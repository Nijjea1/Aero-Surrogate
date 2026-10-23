# Data and model provenance notices

The repository's MIT licence covers the team's source code. It does not
relicense the third-party datasets or reference samples in an artifact bundle.
Preserve these notices when redistributing the bundle or its reference samples.

## DrivAerNet scalar data and model

Source: Mohamed Elrefaie and collaborators, **DrivAerNet / DrivAerNet++**.
Canonical repository: https://github.com/Mohamedelrefaie/DrivAerNet

The source dataset is provided under **Creative Commons Attribution-
NonCommercial 4.0 International (CC BY-NC 4.0)**:
https://creativecommons.org/licenses/by-nc/4.0/

The release scalar checkpoint was trained by the team on 4,165 tabular designs
using the sixteen selected morph parameters plus body-style indicators. The
bundle includes selected derived reference predictions and split identifiers,
not the complete dataset. These reference selections/transformations were made
by the Aero Surrogate team; the dataset authors do not endorse the predictions.
Retain attribution and the source dataset's non-commercial notice with samples.

## DrivAerML field data and model

Source: Neil Ashton, Christoph Mockett, Martin Fuchs and collaborators,
**DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car
External Aerodynamics** (2024).

Dataset and complete author citation:
https://huggingface.co/datasets/neashton/drivaerml
Paper: https://arxiv.org/abs/2408.11969

The source dataset is **Creative Commons Attribution-ShareAlike 4.0
International (CC BY-SA 4.0)**:
https://creativecommons.org/licenses/by-sa/4.0/

If fields are included, `runs/handover/source/run_3.npz` is a team-extracted
reference sample from DrivAerML run 3. It contains transformed surface geometry
and query/reference arrays used to reproduce numerical inference. This derived
sample is distributed under **CC BY-SA 4.0** with the attribution above. The team
performed extraction and normalisation; the original dataset authors do not
endorse the resulting surrogate. Preserve the share-alike terms for redistributed
adaptations of that sample.

The separate field checkpoint was trained by the team on twenty cars, with five
validation and five test cars recorded in its metadata. Its generated viewer
mesh is approximate. Fixed-query parity does not establish full-test accuracy,
viewer orientation, generated-geometry calibration, lift or wall shear.

## Checkpoint provenance and intended use

The bundle's `release-manifest.json` identifies the exact team-produced scalar
and optional field checkpoints, training metadata, source revision and file
checksums. They are supplied for reproducing the hackathon demonstration and
evaluation. Do not substitute these predictions for CFD or physical validation.
The bundle makes no claim that upstream dataset rights are transferred by the
source-code MIT licence. Retain these model/data provenance statements with the
weights and consult the upstream dataset terms for intended reuse.

## Browser assets

The web application vendors **three.js 0.160.0** under its upstream MIT licence;
see `web/vendor/three/LICENSE` and its file-checksum manifest. **IBM Plex** fonts
retain their SIL Open Font License 1.1 notices under `web/vendor/fonts/`.
