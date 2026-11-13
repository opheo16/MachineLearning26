# Laboratory Experiment 3.1: Understanding Data as Tensors

**Course:** Machine Learning  
**Lab Number:** 3.1  
**Lab Title:** Multidimensional Tensors for Images and Videos  
**Student Name:** Priyanshi Mali



---

## 1. Tensor Shape Summary Table

| Data Type | Tensor Shape | Rank (Dimensions) | Meaning |
| :--- | :--- | :---: | :--- |
| **Scalar** | `()` | 0D | A single standalone number (e.g., price = 50) |
| **Vector** | `(features,)` | 1D | A 1D array / flat feature vector |
| **Matrix (Tabular)** | `(samples, features)` | 2D | Traditional table: Rows = samples, Columns = features |
| **Single Grayscale Image** | `(height, width)` | 2D | 2D pixel grid ($0 = \text{black}$, $255 = \text{white}$) |
| **Batch of Grayscale Images**| `(num_images, height, width)` | 3D | Stack/batch of multiple grayscale images |
| **Single RGB Color Image** | `(height, width, channels)` | 3D | 2D grid with 3 color channels ($R, G, B$) |
| **Batch of Color Images** | `(num_images, height, width, channels)` | 4D | Batch of color images for deep learning (CNNs) |
| **Single Video** | `(frames, height, width, channels)` | 4D | Sequence of RGB frames over time |
| **Batch of Videos** | `(num_videos, frames, height, width, channels)` | 5D | Batch of multiple video clips |

---

## 2. Answers to Reflection Questions

### Q1: Why is a grayscale image represented as a 2D tensor?
**Answer:** Each pixel in a grayscale image only measures one thing: brightness/intensity. To locate and display each pixel, we only need 2 coordinates: Row (Height) and Column (Width).

### Q2: Why is an RGB image represented as a 3D tensor?
**Answer:** A color image requires 3 color intensities (Red, Green, Blue) at each pixel to create full color. Therefore, it has 3 dimensions: Height, Width, and 3 Color Channels `(height, width, 3)`.

### Q3: Why is a video represented as a 4D tensor?
**Answer:** A video is a sequence of 3D color images moving through time. The 4th dimension is the Frame/Time index: `(frames, height, width, channels)`.

### Q4: What happens when we flatten an image?
**Answer:** Flattening takes the 2D grid (or 3D tensor) and unrolls it into a single straight 1D line (vector) of numbers. For example, an $8 \times 8$ image becomes a line of 64 numbers.

### Q5: Why can flattening be useful for classical ML?
**Answer:** Classical ML algorithms (like Logistic Regression, SVM, and Random Forests) only accept flat 1D rows of features as inputs.

### Q6: Why can flattening be harmful for image understanding?
**Answer:** Flattening destroys the 2D spatial arrangement. Pixels that were vertical neighbors are pushed far apart in the 1D line, losing shapes, edges, and contours.

### Q7: Why are CNNs better suited for image data than simple flattened input?
**Answer:** CNNs preserve the 2D/3D structure. They slide small filters over the image to naturally recognize local visual patterns like edges, textures, and objects, regardless of where they appear.

### Q8: What does the channel dimension mean in an RGB image?
**Answer:** It represents the specific color wavelength intensity: Channel 0 is Red, Channel 1 is Green, and Channel 2 is Blue.

### Q9: What does the frame dimension mean in a video?
**Answer:** It represents the time axis—each consecutive photo snapshot captured per second in the video sequence.

### Q10: How is tabular data different from image data?
**Answer:** In tabular data, every column has a completely different meaning (e.g., Age, Salary, State). In image data, every number represents the exact same thing: light intensity of a pixel on a grid.