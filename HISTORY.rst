History
=======
0.4.9 (06/15/2020)
------------------

- More syntax updates
- transcribe_vids now works with any directory of videos
- Google Cloud Storage dependencies removed

0.4.8 (06/12/2020)
------------------

- Various syntax updates

0.4.7 (06/10/2020)
------------------

- Updated help docstrings for audio, video, and image modules.
- Added music mood classification to audio module
- videofeats now filters out monochromatic frames.
- Syntax cleanup in video.py

0.4.6 (05/06/2020)
------------------

- Bugfix for transcribe_vids script.

0.4.5 (05/06/2020)
------------------

- Added timing messages to audio_feats script.
- More efficient computation in transcribe_vids script.

0.4.4 (03/13/2020)
------------------

- Fixed issue with searching names in transcribe_vids script.
- Fixed RGB / BGR inconsistencies in face recognition functions.
- Added model option for face_recognition.
- Landmark model for face recognition now uses large model by default.
- File naming updated to be consistent with input data for summarize_vids script.

0.4.3 (12/16/2019)
------------------

- Fixed transcribe_vids variable bug.

0.4.2 (12/07/2019)
------------------

- Fixed image module import issue.

0.4.1 (12/03/2019)
------------------

- Updated documentation for video transcription functions.
- transcribe_vids script output updated.

0.4.0 (12/03/2019)
------------------

- Added face reconition and transcription capabilities.
- Added transcribe_vids script.
- Data folder now distributed with package.

0.3.20 (11/16/2019)
-------------------

- Terminating imtext when trying to read frames now skips to the next video.
- If keyframes.txt already exists, imtext skips to the next video.

0.3.19 (11/16/2019)
-------------------

- Fixed inconsistent shape error when running text detection on different-sized images.

0.3.18 (11/16/2019)
-------------------

- Added exceptions for missing directories.

0.3.17 (11/16/2019)
-------------------

- log.txt deleted at the beginning of imtext script call.
- Better message formatting for standard output printing.

0.3.16 (11/16/2019)
-------------------

- Corrected print issue in imtext.

0.3.15 (11/16/2019)
-------------------

- More debugging.

0.3.14 (11/16/2019)
-------------------

- Import fixes.

0.3.13 (11/16/2019)
-------------------

- Updated imtext.

0.3.12 (11/16/2019)
-------------------

- Faster image text detection with batch processing.
- Fixed repeated model loading issue.
- Updated docstring for Keyframes class and methods.
- Compatibility update with imtext and Keyframes class.

0.3.11 (11/09/2019)
-------------------

- Fixed indexing issue in video module.

0.3.10 (11/07/2019)
-------------------

- Fixed tab issue in imtext.

0.3.9 (11/06/2019)
------------------

- Faster video summarization.
- Fixed issue with feature selection in audio_feats.
- Exception handling for operation timeouts in Vision API calls.

0.3.8 (10/18/2019)
------------------

- Output to stdout in imtext fixed.

0.3.7 (10/18/2019)
------------------

- Fixed UTF-8 encoding issue in imtext.

0.3.6 (10/18/2019)
------------------

- Fixed TypeError issue in writing results.

0.3.5 (10/18/2019)
------------------

- Fixed indexing issue in Image.image_text().

0.3.4 (10/18/2019)
------------------

- Fixed issues with imtext script.
- Image.image_text() now returns image text in the order it appears in the text. 

0.3.3 (10/18/2019)
------------------

- Better imports in __init__.py.

0.3.2 (10/18/2019)
------------------

- Fixed printing in download_models.
- Added python version requirements and updated package dependencies for image module.

0.3.1 (10/17/2019)
------------------

- Minor bugfix in setup.py.

0.3.0 (10/16/2019)
------------------

- Image module added with image text detection.
- imtext script added.
- download_models script added.

0.2.7 (10/12/2019)
------------------

- Changed summarize_vids filenames for keyframes.

0.2.6 (10/09/2019)
------------------

- Changed version requirements for pandas package.

0.2.5 (10/05/2019)
------------------

- audio_feats now returns if no videos found in input directory.

0.2.4 (10/04/2019)
------------------

- Bugfix in audio_feats script.

0.2.3 (10/04/2019)
------------------

- Renamed Spectrogram class to Audio.
- Updated documentation for scripts.
- Added exception handling for scripts.

0.2.2 (10/03/2019)
------------------

- Entry-point issues resolved.

0.2.1 (10/03/2019)
------------------

- Added entry-points for audio_feats, match_vids, and summarize_vids scripts.


0.2.0 (10/03/2019)
------------------

- Updated version requirements for package dependencies.

0.1.0 (10/02/2019)
------------------

- Initial release.