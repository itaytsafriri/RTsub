# RTsubs - Real-Time Hebrew Subtitle Generator

RTsubs is a Python application that generates real-time subtitles from audio input using OpenAI's Whisper model and GPT-3.5 for Hebrew text processing. It's particularly useful for live presentations, meetings, or any scenario where real-time Hebrew subtitles are needed.

## Features

- Real-time audio capture and processing
- Hebrew speech-to-text using Whisper's large model
- Automatic text formatting and punctuation using GPT-3.5
- Right-to-left (RTL) text support for Hebrew
- Configurable subtitle display settings
- Support for various audio input devices

## Prerequisites

- Python 3.7 or higher
- OpenAI API key
- Audio input device (microphone)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RTsubs.git
cd RTsubs
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# On Windows
set OPENAI_API_KEY=your_api_key_here

# On Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the application:
```bash
python rtsubs.py
```

2. The program will:
   - Display available audio input devices
   - Show supported sample rates
   - Start recording and transcribing in real-time
   - Display subtitles in the console

3. Press `Ctrl+C` to stop the application.

## Configuration

You can modify the following parameters in `rtsubs.py`:

- `SAMPLERATE`: Audio sampling rate (default: 16000)
- `CHANNELS`: Number of audio channels (default: 1)
- `BLOCK_SECONDS`: Window size for processing (default: 9 seconds)
- `STEP_SECONDS`: Step size between processing windows (default: 3 seconds)
- `WORDS_PER_LINE`: Maximum words per subtitle line (default: 7)

## Dependencies

- sounddevice: Audio capture
- numpy: Numerical operations
- whisper: Speech recognition
- arabic-reshaper & bidi: RTL text support
- scipy: Signal processing
- openai: GPT-3.5 integration

## License

[Add your chosen license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for Whisper and GPT-3.5
- The open-source community for the various libraries used in this project 