# Neural-Style-Transfer
This repo hosts code for a simple implementation for neural style transfer using tensorflow.


## Setup Instructions

### 1. Setup venv
```bash
python3 -m venv .venv
```


### 2. Activate venv(Unix based distro)
```bash
source .venv/bin/activate
```

### 3. Activate venv(Winudows)
```bash
.\.venv\Scripts\activate
```


### 4. Install requirements
```bash
python3 -m pip install -r requirements.txt
```

## Running the notebook
Open the terminal with the activated venv and run
```bash
jupyter notebook
```
Note: There are some images in the `data/` folder already for quick tests. 

## Testing with a script 
Open the terminal with the activated venv and run
```bash
python .\main.py -c /path/to/content-image -s /path/to/style-image 
```