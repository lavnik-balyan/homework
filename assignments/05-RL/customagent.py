import gymnasium as gym


class Agent:
    """
    Agent Class
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        init variables
        """
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Takes an observation and returns an action.
        """

        x, y, vx, vy, angle, angular_velocity, contact_left, contact_right = observation

        # Heuristic for landing the lander
        if abs(angle) > 0.1:
            # If the angle is too steep, counteract the rotation
            action = 0 if angle < 0 else 2
        elif abs(angular_velocity) > 0.1:
            # If the angular velocity is too high, counteract it
            action = 0 if angular_velocity < 0 else 2
        elif abs(vy) > 0.2 or abs(vx) > 0.1:
            # If the lander is descending too fast or moving horizontally too fast, fire the main engine
            action = 3
        else:
            # Otherwise, do nothing
            action = 1

        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Takes an observation, a reward, a boolean indicating whether the episode has terminated,
        and a boolean indicating whether the episode was truncated.
        """

        pass
