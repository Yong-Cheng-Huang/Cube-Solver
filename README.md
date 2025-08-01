# Rubik’s Cube Solver

A Rubik’s Cube scanning and solving system using OpenCV and the Kociemba algorithm.

## Features

- **Six-face Scanning**: Scan all six faces of the Rubik’s Cube in URFDLB order
- **Color Detection**: Automatically detects six colors — white, red, orange, yellow, green, and blue
- **Smart Grid Sorting**: Automatically sorts 9 detected squares into a 3x3 grid
- **Kociemba Solving**: Uses the Kociemba two-phase algorithm to solve the cube
- **Real-time Display**: Shows live scanning status and detection results

## Demo
<img width="500" alt="截圖 2025-08-01 09 40 59" src="https://github.com/user-attachments/assets/2e669fa1-5389-4b5d-89bf-43285309ddee" />
<img width="500" alt="截圖 2025-08-01 20 22 36" src="https://github.com/user-attachments/assets/0f6b302e-e749-44ef-950b-6e4d1f1d662d" />
<img width="500" alt="截圖 2025-08-01 20 20 22" src="https://github.com/user-attachments/assets/888913c3-f243-41bf-bb84-74eb35743117" />


## Installation Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the Program**:
   ```bash
   python main.py
   ```

2. **Scanning Process**:
   - Press `s` to enter scanning mode
   - Point one face of the cube toward the camera
   - Ensure 9 squares are detected (indicated with green boxes)
   - Press `n` to confirm the current face and proceed to the next
   - Repeat until all 6 faces are scanned

3. **Solving the Cube**:
   - After scanning all faces, press `r` to solve
   - The system will display the Kociemba solution sequence

4. **Exit the Program**:
   - Press `q` to quit

## Color Mapping

| Color  | Kociemba Code | Description     |
|:------:|:-------------:|:---------------:|
| White  | U             | Up (top face)   |
| Red    | R             | Right face      |
| Green  | F             | Front face      |
| Orange | L             | Left face       |
| Blue   | B             | Back face       |
| Yellow | D             | Down (bottom face) |

## Cube Annotation

```
             |************|
             |*U1**U2**U3*|
             |************|
             |*U4**U5**U6*|
             |************|
             |*U7**U8**U9*|
 ************|************|************|************
 *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*
 ************|************|************|************
 *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*
 ************|************|************|************
 *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*
 ************|************|************|************
             |*D1**D2**D3*|
             |************|
             |*D4**D5**D6*|
             |************|
             |*D7**D8**D9*|
             |************|
```

## Scanning Order

The program scans the cube in the following order:
1. **U** (Up) - Top face
2. **R** (Right) - Right face
3. **F** (Front) - Front face
4. **D** (Down) - Bottom face
5. **L** (Left) - Left face
6. **B** (Back) - Back face

## Notes

- Make sure the lighting is sufficient and avoid shadows, which can affect color detection.

## Technical Details

- **Image Processing**: Uses OpenCV for edge detection and contour recognition
- **Color Classification**: Uses HSV color space thresholds to classify colors
- **Square Sorting**: Uses a grid-based algorithm to sort detected squares into a 3x3 layout
- **Solving Algorithm**: Implements Kociemba’s two-phase algorithm

## Troubleshooting

1. **No squares detected**: Check lighting conditions and ensure the cube face is fully visible
2. **Incorrect color detection**: Adjust HSV thresholds in the `classify_color` function
3. **Solving failure**: Verify that all 54 squares were scanned and classified correctly
