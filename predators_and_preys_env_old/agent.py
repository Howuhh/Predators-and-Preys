import abc


class PredatorAgent:
    @abc.abstractmethod
    def act(self, state_dict):
        pass


class PreyAgent:
    @abc.abstractmethod
    def act(self, state_dict):
        pass

