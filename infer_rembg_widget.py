from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_rembg.infer_rembg_process import InferRembgParam
from infer_rembg.core import REMBG_MODELS

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferRembgWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferRembgParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model name")
        for model_name in REMBG_MODELS:
            self.combo_model.addItem(model_name)

        self.combo_model.setCurrentText(self.parameters.model_name)

        # Alpha matting
        self.check_matting = pyqtutils.append_check(self.grid_layout, "Alpha matting", self.parameters.alpha_matting)

        # Mask post-process
        self.check_mask_post_proc = pyqtutils.append_check(self.grid_layout,
                                                           "Post-process mask",
                                                           self.parameters.post_process_mask)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.alpha_matting = self.check_matting.isChecked()
        self.parameters.post_process_mask = self.check_mask_post_proc.isChecked()

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferRembgWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_rembg"

    def create(self, param):
        # Create widget object
        return InferRembgWidget(param, None)
