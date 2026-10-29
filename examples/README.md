# Sample images (download pending)

Planned count: 20. Current count: 0. The official download could not be
completed in the packaging environment; no images have been bundled yet.

Selected source: Microsoft's Face Synthetics public dataset. These are
synthetic faces, not CASIA-WebFace, LFW or TinyFace benchmark images.

- Dataset: https://github.com/microsoft/FaceSynthetics
- Official 100-image sample: https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2021/dataset_100.zip
- Data terms: https://github.com/microsoft/FaceSynthetics/blob/main/LICENSE.txt

The data uses the Research Use of Data Agreement v1.0, not MIT. It permits
redistribution subject to its terms, restricts use to non-commercial
computational research, and requires downstream recipients to retain
attribution and be bound to the same agreement. Before bundling images,
include the complete upstream agreement and all accompanying attribution.
These data terms must remain separate from any future HEFDD code license.

On successful download, select the first 20 rendered face PNGs in sorted
filename order, excluding segmentation masks. Preserve original image bytes
and filenames, and record their source archive and SHA256 hashes. Do not
bundle landmark or segmentation annotations unless separately needed.

Dataset reference: Erroll Wood, Tadas Baltrusaitis, Charlie Hewitt, Sebastian
Dziadzio, Thomas J. Cashman and Jamie Shotton. Fake It Till You Make It:
Face Analysis in the Wild Using Synthetic Data Alone. ICCV, 2021.

Samples demonstrate input formats only. They are not identity-labelled
verification pairs, an accuracy test set, or evidence of HEFDD performance.
