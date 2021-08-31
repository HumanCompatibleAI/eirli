.. interpret:


Interpreting Results
====================

Our current implementation contains some helpful tools for further
analyzing the learned policies. The first helps to generate clusters of
encoder outputs, and the second analyzes a policy with interpretability
algorithms provided in the `Captum <https://captum.ai>`__ package.

Generating Clusters
-------------------

The code to generate clusters is provided in
``analysis/clusters.ipynb``. This notebook enables you to get
representation encodings from saved policies, and visualize the clusters
using Principal component analysis (PCA) or t-distributed Stochastic
Neighbor Embedding (t-SNE) in Tensorboard. What it does is to generate
and save representations into a ``runs`` directory, and visualizing it
is as simple as running two lines:

.. code:: 

   %load_ext tensorboard
   %tensorboard --logdir=runs

Interpreting Policies
---------------------

Sometimes it can be helpful to visualize to what extent a policy relies
on certain regions of the state input, and this can be done by
algorithms provided by `Captum <https://captum.ai/>`__. In
``scripts/interpret.py`` we incorporate three Primary Attribution
methods: `Saliency <https://captum.ai/api/saliency.html>`__, `Integrated
Gradients <https://captum.ai/api/integrated_gradients.html>`__, and
`DeepLift <https://captum.ai/api/deep_lift.html>`__.

To run the script, you need to specify a few parameters:

-  ``encoder_path``: The path leading to the saved encoder you want to
   interpret.

-  ``chosen_algo`` (Optional): The interpretation algorithm you want to
   run. We currently support one of
   ``['saliency', 'integrated_gradient', 'deep_lift']``, with
   ``'integrated_gradient'`` being the default value.

-  ``length`` (Optional): The number of images you want to interpret.
   This will make the policy interpret the first ``length`` images from
   the dataset. The default value is ``2``.

-  ``save_video`` (Optional): Whether to save the interpreted ``length``
   images as a video. The default value is ``False``.

-  ``save_image`` (Optional): Whether to save ``length`` images to a
   local disk. The default value is ``True``.

-  ``device`` (Optional): Specify the device you want to run. By
   default, it will use CUDA if a GPU is available, and use CPU
   otherwise.

The benchmark and dataset to be tested on by default depends on
``env_cfg_defaults`` in ``envs/config.py``. If you want to specify them
on-the-fly, you can set the values when you call this file. Below is an
example to run the interpretation on ``Procgen``'s ``Coinrun``
environment:

.. code:: shell

   CUDA_VISIBLE_DEVICES=1 python ./src/il_representations/scripts/interpret.py with \
   	encoder_path=${path_to_encoder} \
   	save_video=True \
   	save_image=True \
   	chosen_algo=saliency \
   	length=1000 \
   	env_cfg.benchmark_name=procgen \
   	env_cfg.task_name=coinrun
