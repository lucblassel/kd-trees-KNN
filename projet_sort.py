x = [[1,2],[5,7],[-1,2]]
def shellSort(array,dim):
 "Shell sort using Shell's (original) gap sequence: n/2, n/4, ..., 1."
 gap = len(array) // 2
 # loop over the gaps
 while gap > 0:
     # do the insertion sort
     for i in range(gap, len(array)):
         val = array[i][dim]
         sub_array = array[i]
         j = i
         while j >= gap and array[j - gap][dim] > val:
             array[j] = array[j - gap]
             j -= gap
         array[j] = sub_array
     gap //= 2