from .viewer import TopologicalAbstractor, SCCModularityAbstractor
from .bpmn_adapter import bpmn_to_dfg_graph


class BPMNTopologicalAbstractor(TopologicalAbstractor):
    def __init__(self, bpmn_model, labeler):
        self.log = None
        self.labeler = labeler
        self.history = []

        self.G = bpmn_to_dfg_graph(bpmn_model)
        self.history.append(self.get_current_state())
        self.run_abstraction()


class BPMNSCCModularityAbstractor(SCCModularityAbstractor):
    def __init__(self, bpmn_model, labeler):
        self.log = None
        self.labeler = labeler
        self.history = []

        self.G = bpmn_to_dfg_graph(bpmn_model)
        self.history.append(self.get_current_state())
        self.run_abstraction()
