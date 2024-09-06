<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_rembg</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_rembg">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_rembg">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_rembg/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_rembg.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm proposes inference on various models to remove image background.
It is based on the [rembg](https://github.com/danielgatis/rembg) library (CPU version only).

![Illustration image](https://raw.githubusercontent.com/Ikomia-hub/infer_rembg/main/images/illustration.jpg)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
# Init your workflow
wf = Workflow()    

# Add the real_esrgan algorithm
algo = wf.add_task(name = 'infer_rembg', auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(1).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- model_name (str): name of the model. Default: **u2net**
- post_process_mask (bool): enable/disable mask post processing
- alpha_matting (bool): enable/disable alpha matting
- alpha_matting_fg_threshold (int): foreground threshold. Default: **240**
- alpha_matting_bg_threshold (int): background threshold. Default: **10**
- alpha_matting_erode_size (int): kernel size for erosion. Default: **10**

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
rembg = wf.add_task(name="infer_rembg", auto_connect=True)

rembg.set_parameters({
    "model_name": "isnet-general-use",
    "post_process_mask": "False",
    "alpha_matting": "True",
    "alpha_matting_fg_threshold": "240",
    "alpha_matting_bg_threshold": "10",
    "alpha_matting_erode_size": "7",
})

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Inspect your results
display(rembg.get_input(0).get_image())
display(rembg.get_output(1).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_rembg", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_portrait.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
