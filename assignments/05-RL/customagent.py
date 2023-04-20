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

        x, y, dx, dy, angle, angular_speed, l_leg, r_leg = observation

        # Rule 1: If the lander is tilting left, fire the right engine
        if angle < -0.1:
            return 2
        # Rule 2: If the lander is tilting right, fire the left engine
        if angle > 0.1:
            return 1
        # Rule 3: If the lander is falling too fast, fire the main engine
        if dy < -0.3:
            return 3
        # If no rules apply, do nothing
        return 0

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
