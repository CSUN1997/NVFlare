from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.apis.controller_spec import OperatorMethod, TaskOperatorKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Task
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.security.logging import secure_format_exception


class HierarchicalScatterAndGather(ScatterAndGather):
    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        try:
            self.log_info(fl_ctx, "Beginning ScatterAndGather training phase.")
            self._phase = AppConstants.PHASE_TRAIN

            fl_ctx.set_prop(AppConstants.PHASE, self._phase, private=True, sticky=False)
            fl_ctx.set_prop(AppConstants.NUM_ROUNDS, self._num_rounds, private=True, sticky=False)
            self.fire_event(AppEventType.TRAINING_STARTED, fl_ctx)

            if self._current_round is None:
                self._current_round = self._start_round
            while self._current_round < self._start_round + self._num_rounds:

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, f"Round {self._current_round} started.")
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self._current_round, private=True, sticky=True)
                self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

                # Create train_task
                data_shareable: Shareable = self.shareable_gen.learnable_to_shareable(self._global_weights, fl_ctx)
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self._current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self._num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self._current_round)

                operator = {
                    TaskOperatorKey.OP_ID: self.train_task_name,
                    TaskOperatorKey.METHOD: OperatorMethod.BROADCAST,
                    TaskOperatorKey.TIMEOUT: self._train_timeout,
                    TaskOperatorKey.AGGREGATOR: self.aggregator_id,
                }

                train_task = Task(
                    name=self.train_task_name,
                    data=data_shareable,
                    operator=operator,
                    props={},
                    timeout=self._train_timeout,
                    before_task_sent_cb=self._prepare_train_task_data,
                    result_received_cb=self._process_train_result,
                )

                self.broadcast_and_wait(
                    task=train_task,
                    min_responses=self._min_clients,
                    wait_time_after_min_received=self._wait_time_after_min_received,
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.log_info(fl_ctx, "Start aggregation.")
                self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
                aggr_result = self.aggregator.aggregate(fl_ctx)
                fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
                self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
                self.log_info(fl_ctx, "End aggregation.")

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                self.fire_event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE, fl_ctx)
                self._global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)
                fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._global_weights, private=True, sticky=True)
                fl_ctx.sync_sticky()
                self.fire_event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE, fl_ctx)

                if self._check_abort_signal(fl_ctx, abort_signal):
                    return

                if self.persistor:
                    if (
                        self._persist_every_n_rounds != 0
                        and (self._current_round + 1) % self._persist_every_n_rounds == 0
                    ) or self._current_round == self._start_round + self._num_rounds - 1:
                        self.log_info(fl_ctx, "Start persist model on server.")
                        self.fire_event(AppEventType.BEFORE_LEARNABLE_PERSIST, fl_ctx)
                        self.persistor.save(self._global_weights, fl_ctx)
                        self.fire_event(AppEventType.AFTER_LEARNABLE_PERSIST, fl_ctx)
                        self.log_info(fl_ctx, "End persist model on server.")

                self.fire_event(AppEventType.ROUND_DONE, fl_ctx)
                self.log_info(fl_ctx, f"Round {self._current_round} finished.")
                self._current_round += 1

                # need to persist snapshot after round increased because the global weights should be set to
                # the last finished round's result
                # if self._snapshot_every_n_rounds != 0 and self._current_round % self._snapshot_every_n_rounds == 0:
                #     self._engine.persist_components(fl_ctx, completed=False)

            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, "Finished ScatterAndGather Training.")
        except Exception as e:
            error_msg = f"Exception in ScatterAndGather control_flow: {secure_format_exception(e)}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(error_msg, fl_ctx)
