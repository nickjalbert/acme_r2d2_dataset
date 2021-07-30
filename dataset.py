# TODO - update reqs
import agentos
from acme import datasets
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.adders import reverb as adders
import reverb


from dm_env import TimeStep
from dm_env import StepType


class ReverbDataset(agentos.Dataset):
    @classmethod
    def ready_to_initialize(cls, shared_data):
        return "environment_spec" in shared_data and "network" in shared_data

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BURN_IN_LENGTH = 2
        TRACE_LENGTH = 10
        SEQUENCE_LENGTH = BURN_IN_LENGTH + TRACE_LENGTH + 1
        # TODO - remove params that aren't actually shared
        self.shared_data["burn_in_length"] = BURN_IN_LENGTH
        self.shared_data["trace_length"] = TRACE_LENGTH
        self.shared_data["sequence_length"] = SEQUENCE_LENGTH
        self.shared_data["replay_period"] = 40
        self.shared_data["batch_size"] = 32
        self.shared_data["prefetch_size"] = tf.data.experimental.AUTOTUNE
        PRIORITY_EXPONENT = 0.6
        self.shared_data["priority_exponent"] = PRIORITY_EXPONENT
        self.shared_data["max_replay_size"] = 500

        initial_state = self.shared_data["network"].initial_state(1)
        extra_spec = {
            "core_state": tf2_utils.squeeze_batch_dim(initial_state),
        }
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(PRIORITY_EXPONENT),
            remover=reverb.selectors.Fifo(),
            max_size=self.shared_data["max_replay_size"],
            rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1),
            signature=adders.SequenceAdder.signature(
                self.shared_data["environment_spec"],
                extra_spec,
                sequence_length=self.shared_data["sequence_length"],
            ),
        )

        # NB - must save ref to server or it gets killed
        self.reverb_server = reverb.Server([replay_table], port=None)
        address = f"localhost:{self.reverb_server.port}"
        self.shared_data["dataset_address"] = address

        # Component to add things into replay.
        adder = adders.SequenceAdder(
            client=reverb.Client(self.shared_data["dataset_address"]),
            period=self.shared_data["replay_period"],
            sequence_length=self.shared_data["sequence_length"],
        )
        self.shared_data["adder"] = adder  # TODO - remove adder from POLICY

        # The dataset object to learn from.
        dataset = datasets.make_reverb_dataset(
            server_address=address,
            batch_size=self.shared_data["batch_size"],
            prefetch_size=self.shared_data["prefetch_size"],
        )

        self.shared_data["dataset"] = dataset

    # https://github.com/deepmind/acme/blob/master/acme/agents/tf/actors.py#L164
    def add(self, prev_obs, action, curr_obs, reward, done, info):
        if action is None:  # No action -> first step
            timestep = TimeStep(StepType.FIRST, None, None, curr_obs)
            self.shared_data["adder"].add_first(timestep)
        else:
            if done:
                timestep = TimeStep(
                    StepType.LAST,
                    reward,
                    self.shared_data["discount"],
                    curr_obs,
                )
            else:
                timestep = TimeStep(
                    StepType.MID,
                    reward,
                    self.shared_data["discount"],
                    curr_obs,
                )

            # FIXME - hacky way to push observation counts
            if "num_observations" not in self.shared_data:
                self.shared_data["num_observations"] = 0
            self.shared_data["num_observations"] += 1

            # FIXME - hacky way to push recurrent state
            if self.shared_data["_prev_state"] is not None:
                numpy_state = tf2_utils.to_numpy_squeeze(
                    self.shared_data["_prev_state"]
                )
                self.shared_data["adder"].add(
                    action, timestep, extras=(numpy_state,)
                )

            else:
                # self.shared_data['adder'].add(action, timestep)
                raise Exception("Recurrent state not available")
