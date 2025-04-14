# Sentinel-2 AI Processor

This application allows you to browse and view Sentinel-2 satellite imagery from the Copernicus Data Space Ecosystem. You can search by coordinates, select various landscapes, and filter by cloud cover.


## Installation

1. Clone the repository:

```bash
git clone git@github.com:sebastien-tetaud/sentinel-2-ai-processor.git
cd sentinel-2-ai-processor
```

2. Create and activate a conda environment:

```bash
conda create -n ai_processor python==3.13.2
conda activate ai_processor
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your credentials by creating a `.env` file in the root directory with the following content:

```bash
touch .env
```
then:

```
ACCESS_KEY_ID=username
SECRET_ACCESS_KEY=password
```
