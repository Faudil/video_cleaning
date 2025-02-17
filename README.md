
# Video Cleaner  
  
## Overview  
  
The `video_cleaner` program is designed to clean and order frames from a video file. It reads frames from a video, filters out outlier frames, orders the remaining frames, and then saves the cleaned video and rejected frames.  
  
## Features  
  
- **Frame Filtering**: Removes outlier frames based on clustering.  
- **Frame Ordering**: Orders the remaining frames for a coherent video sequence.  
- **Multiple Encoders**: Supports different feature extraction methods (`vit`, `orb`, `dino`).  
- **Output Options**: Saves the cleaned video and rejected frames to specified locations.  
    
## Installation  
  
1. Clone the repository:  
  ``` bash  
  git clone <repository-url>  
 cd video_cleaner ```  
2. Install the required packages:  
   ```bash  
  pip install -r requirements.txt  
 ```  

## Usage  
  
To use the `video_cleaner`, run the script from the command line with the appropriate arguments:  
  
```bash  
python video_cleaner.py <filename> [-e <encoder>] [-o <output>] [-d <dir>]
```  
  
### Arguments  
  
- `filename`: Path to the video file to be processed.  
- `-e, --encoder`: Specify the encoder to use (`vit`, `orb`, `dino`). Default is `vit`.  
- `-o, --output`: Path to save the cleaned video. Default is `cleaned_video.mp4`.  
- `-d, --dir`: Directory to save rejected frames. Default is `rejected_frames`.  
- `-v, --verbose`: Enable verbose output.  
  
### Example  
  
```bash  
python video_cleaner.py input_video.mp4 -e orb -o cleaned_output.mp4 -d rejected_frames_dir -v  
```  
  
## How It Works  
  
1. **Load Frames**: Reads all frames from the input video.  
2. **Feature Extraction**: Extracts features from each frame using the specified encoder.  
3. **Filter Frames**: Filters out outlier frames based on clustering.  
4. **Order Frames**: Orders the remaining frames for a coherent sequence.  
5. **Save Results**: Saves the cleaned video and rejected frames to the specified locations.
