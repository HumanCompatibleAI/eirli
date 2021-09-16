.. _reproduction:


Reproduction of Benchmark Paper Experiments 
===========================================

To reproduce our results, you will first need to download our `demonstration
dataset <https://berkeley.app.box.com/s/8yo3yyyh0h2e1ay5iehbnyg4g0cm0lpe>`_. You
should unzip the split archive with 7zip, then place the ``data/`` directory in
the root of our repository (so, for instance, there will be ``finger-spin``
demos at
``/path/to/our-repo.git/data/processed/demos/dm_control/finger-spin/demos.tgz``.

MAGICAL 
-------
Pretraining experiments can be reproduced with `submit_expts_2021_08_19_magical_pretrain.sh <https://github.com/HumanCompatibleAI/il-representations/blob/sam-new-vis/cloud/submit_expts_2021_08_19_magical_pretrain.sh>`_

Joint training experiments can be reproduced with the script `submit_expts_2021_08_20_magical_jt.sh <https://github.com/HumanCompatibleAI/il-representations/blob/sam-new-vis/cloud/submit_expts_2021_08_20_magical_jt.sh>`_

DMC / ProcGen 
-------------
Pretraining experiments can be reproduced with the script `submit_expts_2021_08_27_orig_neurips_bt_procgen_dmc_pretrain_expts.sh <https://github.com/HumanCompatibleAI/il-representations/blob/sam-new-vis/cloud/submit_expts_2021_08_27_orig_neurips_bt_procgen_dmc_pretrain_expts.sh>`_

Joint training experiments can be reproduced with the script `submit_expts_2021_08_27_orig_neurips_bt_procgen_dmc_jt_expts.sh <https://github.com/HumanCompatibleAI/il-representations/blob/sam-new-vis/cloud/submit_expts_2021_08_27_orig_neurips_bt_procgen_dmc_jt_expts.sh>`_
