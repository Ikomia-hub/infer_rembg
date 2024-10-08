import copy
from ikomia import core, dataprocess
from ikomia.utils import strtobool
from rembg import new_session
from infer_rembg.core import run_rembg


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferRembgParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # GPU is not ready for the moment for the following reasons:
        # - Weird behavior in rembg when we set ORT providers -> available providers (even not functional) are
        #   systematically added to the list and let ORT choose the one to use
        # - Bug in dependency management in rembg: rembg[gpu] install onnxruntime and onnxruntime-gpu side by side
        #   and CPU is then the only functional provider
        # - Issue in onnxruntime with cudnn linking. This requires a system-wide cuda/cudnn installation which is not
        #   possible for automatic deployment with Ikomia Scale. An issue is opened in onnx repo.
        # self.gpu = False
        self.model_name = "u2net"
        self.alpha_matting = False
        self.alpha_matting_fg_threshold = 240
        self.alpha_matting_bg_threshold = 10
        self.alpha_matting_erode_size = 10
        self.post_process_mask = False

    def set_values(self, params):
        # Set parameters values from Ikomia Studio or API
        # Parameters values are stored as string and accessible like a python dict
        # self.gpu = strtobool(params["gpu"])
        self.model_name = params["model_name"]
        self.alpha_matting = strtobool(params["alpha_matting"])
        self.alpha_matting_fg_threshold = int(params["alpha_matting_fg_threshold"])
        self.alpha_matting_bg_threshold = int(params["alpha_matting_bg_threshold"])
        self.alpha_matting_erode_size = int(params["alpha_matting_erode_size"])
        self.post_process_mask = strtobool(params["post_process_mask"])

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        params = {
            # "gpu": str(self.gpu),
            "model_name": self.model_name,
            "alpha_matting": str(self.alpha_matting),
            "post_process_mask": str(self.post_process_mask),
            "alpha_matting_fg_threshold": str(self.alpha_matting_fg_threshold),
            "alpha_matting_bg_threshold": str(self.alpha_matting_bg_threshold),
            "alpha_matting_erode_size": str(self.alpha_matting_erode_size),
        }
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferRembg(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.add_output(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferRembgParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        current_param = self.get_param_object()
        self.model_name = current_param.model_name
        self.session = None
        # self.gpu = current_param.gpu

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def _get_sam_input_prompt(self):
        graphics_input = self.get_input(1)
        items = graphics_input.get_items()
        prompt = []

        for item in items:
            if item.get_type() == core.GraphicsItem.POINT:
                prompt.append({
                    "type": "point",
                    "data": [item.point.x, item.point.y],
                    "label": 1  # foreground
                })

        return prompt

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()
        # if self.session is None or self.model_name != param.model_name or self.gpu != param.gpu:
        if self.session is None or self.model_name != param.model_name:
            # self.gpu = param.gpu
            self.model_name = param.model_name
            # providers = ["CUDAExecutionProvider"] if self.gpu else ["CPUExecutionProvider"]
            providers = ["CPUExecutionProvider"]
            self.session = new_session(self.model_name, providers=providers)

        img_input = self.get_input(0)
        src_image = img_input.get_image()

        if self.model_name == "sam":
            prompt = self._get_sam_input_prompt()
            if len(prompt) == 0:
                raise RuntimeError("SAM model requires input points")

            mask, output_img = run_rembg(session=self.session,
                                         src_image=src_image,
                                         post_process_mask=param.post_process_mask,
                                         alpha_matting=param.alpha_matting,
                                         alpha_matting_fg_threshold=param.alpha_matting_fg_threshold,
                                         alpha_matting_bg_threshold=param.alpha_matting_bg_threshold,
                                         alpha_matting_erode_size=param.alpha_matting_erode_size,
                                         sam_prompt=prompt)
        else:
            mask, output_img = run_rembg(session=self.session,
                                         src_image=src_image,
                                         post_process_mask=param.post_process_mask,
                                         alpha_matting=param.alpha_matting,
                                         alpha_matting_fg_threshold=param.alpha_matting_fg_threshold,
                                         alpha_matting_bg_threshold=param.alpha_matting_bg_threshold,
                                         alpha_matting_erode_size=param.alpha_matting_erode_size)

        # Get output :
        task_output = self.get_output(0)
        task_output.set_image(mask)
        task_output = self.get_output(1)
        task_output.set_image(output_img)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferRembgFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_rembg"
        self.info.short_description = "Remove background with rembg library"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Background"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Daniel Gatis"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "MIT"

        # Ikomia API compatibility
        # self.info.min_ikomia_version = "0.11.1"
        # self.info.max_ikomia_version = "0.11.1"

        # Python compatibility
        # self.info.min_python_version = "3.10.0"
        # self.info.max_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = ""

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_rembg"
        self.info.original_repository = "https://github.com/danielgatis/rembg"

        # Keywords used for search
        self.info.keywords = "remove,background,alpha,matting"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER

        # Algorithms tasks: CLASSIFICATION, COLORIZATION, IMAGE_CAPTIONING, IMAGE_GENERATION,
        # IMAGE_MATTING, INPAINTING, INSTANCE_SEGMENTATION, KEYPOINTS_DETECTION,
        # OBJECT_DETECTION, OBJECT_TRACKING, OCR, OPTICAL_FLOW, OTHER, PANOPTIC_SEGMENTATION,
        # SEMANTIC_SEGMENTATION or SUPER_RESOLUTION
        self.info.algo_tasks = "IMAGE_MATTING"

    def create(self, param=None):
        # Create algorithm object
        return InferRembg(self.info.name, param)
