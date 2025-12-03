import numpy as np

def mask_and_classify_scores(arr):
   if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 4:
       return None
   cleaned = arr.copy()
    
   cleaned[cleaned < 0] = 0
   cleaned[cleaned > 100] = 100
                 
   levels = np.zeros(cleaned.shape, dtype=int)
                       
   for i in range(cleaned.shape[0]):
        for j in range(cleaned.shape[1]):
            value = cleaned[i, j]
            if value < 40:
                levels [i, j]= 0
            elif 40 <= value < 70:
                levels[i, j]= 1
            else:
                levels[i, j]= 2
    
   n = arr.shape[0]
   row_pass_counts = np.zeros(n, dtype=int)
    
   for i in range(cleaned.shape[0]):
         count = 0
         for value in cleaned[i]: 
             if value >= 50:
                 count += 1  
                 row_pass_counts[i] = count
    
   return cleaned, levels, row_pass_counts
        