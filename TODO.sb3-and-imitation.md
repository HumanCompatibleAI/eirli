# Notes on imitation/SB3 update (2021-10-18)

Revision of imitation library that we used for experiments:
`08585bc90108fbad02359ae2e32c40352c369df8`

Things that changed between revision 08585bc and origin/image-env-changes:

- Merged in a bunch of commits from master (probably irrelevant).
- Reverted the commit that gave us back inconsistent augmentations.
  Augmentations are consistent at origin/image-env-changes. Ideally we should
  make this configurable!
- Adds a test to make sure augmentations _are_ consistent (again, we should
  update if we decide to make this a configurable parameter).
- A bunch of test auto-skip stuff.

What I know I need to do:

- Add an option to enable inconsistent augmentations.
- Update the SB3 dependency to point to the latest commit.
- Deal with any changes that have been made in parallel in master. Specifically:
  - Figure out whether `record_mean` approach in BC is the right one (I added a
    FIXME there).

For ILR:

- Update the Dockerfile construction code (and other scripts) to download a
  valid MuJoCo key directly, rather than copying it from the host system.
- At the moment there is a weird line in bc.py that tries to "undo the
  preprocessing we did before applying augmentations", on the theory that SB3
  policies forcibly apply preprocessing anyway. I should check whether that is
  still the case.
- Deal with breakage created by having a non-global logger for imitation.
- Switch from `policy_class`/`policy_kwargs` in BC constructor to system where
  we pass in policy directly.
- Change `expert_data` kwarg in BC to `demonstrations`.
- Deal with the fact that [`DiscrimNet` no longer
  exists](https://github.com/HumanCompatibleAI/imitation/pull/354/files).

Revision of SB3 that we used for experiments:
`ad902dd5f9d4afeef347897cb04c48a19a659ca3`
