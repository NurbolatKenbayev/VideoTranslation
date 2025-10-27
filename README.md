# Video Translation Pipeline

The system is implemented in Python using FastAPI, exposing endpoints to process the video. 

The pipeline takes a video file (with English audio) and its English transcript as input, and produces a new video with German audio dubbed in sync with the visuals. Each major task in the workflow is handled by a separate agent (module), enabling clear separation of concerns and easy future extension and iterative improvement. 

The primary agents include:
-
- 
- 

There is FastAPI endpoint to make a whole video translation via orchestration over agents, but also endpoints that allow you to expose individual agent independently.

This modular design facilitates debugging (by inspecting intermediate outputs like transcripts or translated text) and future improvements (such as supporting additional languages) with minimal changes to the overall architecture.


```
brew install ffmpeg
```