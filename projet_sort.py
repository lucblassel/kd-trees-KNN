"""
author: Romain Gautron
Sorting functions for KD-tree implementation of KNN
"""

def shell_sort(array,dim):
 "Shell sort using Shell's (original) gap sequence: n/2, n/4, ..., 1."
 gap = len(array) // 2
 # loop over the gaps
 while gap > 0:
     # do the insertion sort
     for i in range(gap, len(array)):
         val = array[i][dim]
         subArray = array[i]
         j = i
         while j >= gap and array[j - gap][dim] > val:
             array[j] = array[j - gap]
             j -= gap
         array[j] = subArray
     gap //= 2

def quicksort(array, start, end, axis):
    """
    sort array of k-dimensional points along specified axis using quicksort
    """
    if start < end:
        pivot = partition(array,start,end,axis)
        quicksort(array,start,pivot-1,axis)
        quicksort(array,pivot+1,end,axis)

def partition(array, start, end, axis):
    """
    partitions along specified axis for dimension-wise quicksort
    """
    pivot = array[start][axis]
    left = start+1
    right = end
    done = False
    while not done:
        while left <= right and array[left][axis] <= pivot:
            left = left+1
        while array[right][axis] >= pivot and right >= left:
            right = right-1
        if right < left:
            done = True
        else:
            temp = array[left]
            array[left] = array[right]
            array[right] = temp
    temp = array[start]
    array[start] = array[right]
    array[right] = temp
    return right

def main():
    array =[[1, 3],[1, 8], [2, 2], [2, 10], [3, 6], [4, 1], [5, 4], [6, 8], [7, 4], [7, 7], [8, 2], [8, 5],[9, 9]]
    print(array)
    quicksort(array,0,len(array)-1,1)
    print(array)

if __name__=='__main__':
    main()
