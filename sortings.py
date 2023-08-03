import random


# bubble sort O(n^2)
def bubble_sort(arr: [float]) -> None:
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            # print(f"arr = {arr}")
            # print(f"{arr[j]} > {arr[j+1]} (j = {j}) (i = {i})???")
            # print("----")
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


# insertion sort O(n^2) better if data is "mostly sorted"
def insertion_sort(arr: [float]) -> None:
    for i in range(1, len(arr)):
        curr = arr[i]
        j: int = i - 1
        while j > -1 and arr[j] > curr:
            # print(f"arr = {arr}")
            # print(f"arr[j+1] = {arr[j+1]} arr[j] = {arr[j]} j = {j}")
            arr[j + 1] = arr[j]
            # print(f"arr = {arr}")
            j -= 1
        arr[j + 1] = curr
        # print(f"arr = {arr}")


# selection sort O(n^2)
def selection_sort(arr: [float]) -> None:
    for i in range(len(arr)):
        min_ind: int = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_ind]:
                min_ind = j
        if min_ind != i:
            arr[min_ind], arr[i] = arr[i], arr[min_ind]


# merge sort O(n log n) (more reliable than qsort)
def merge_sort(arr: [float]) -> [float]:
    if len(arr) < 2: return arr
    mid = len(arr) // 2
    left_arr = arr[:mid]
    right_arr = arr[mid:]
    # print(f"left_arr = {left_arr}")
    # print(f"right_arr = {right_arr}")

    left_arr = merge_sort(left_arr)
    right_arr = merge_sort(right_arr)

    def merge(left, right):
        res_arr = []
        while len(left) and len(right):
            if left[0] < right[0]:
                res_arr.append(left_arr.pop(0))
            else:
                res_arr.append(right_arr.pop(0))
        # print(f"merging phase1 = {res_arr}")

        while len(left):
            res_arr.append(left_arr.pop(0))

        while len(right):
            res_arr.append(right_arr.pop(0))
        # print(f"merging phase2 = {res_arr}")

        return res_arr

    return merge(left_arr, right_arr)


# quick sort not the best implementation
def quick_sort_bad(arr: [float]) -> [float]:
    if len(arr) < 2: return arr

    pivot = arr[len(arr) // 2]

    left_arr = [i for i in arr if i < pivot]
    middle_arr = [i for i in arr if i == pivot]
    right_arr = [i for i in arr if i > pivot]

    return quick_sort_bad(left_arr) + middle_arr + quick_sort_bad(right_arr)


# quick sort O(n log n) (depends a lot on pivot)
def quick_sort_good(array: [float], low: int = 0, high: int = None) -> None:
    if high is None: high = len(array) - 1

    def partition(arr: [float], low: int, high: int) -> int:
        # print(arr) ---

        # Choose the rightmost element as pivot
        pivot = arr[high]

        # print(f"pivot = {pivot}") ---

        # Pointer for greater element
        i = low - 1

        # Traverse through all elements
        # compare each element with pivot
        for j in range(low, high):
            # print(f"j = {j}") ---
            if arr[j] <= pivot:
                # print(f"{arr[j]} <= {pivot}") ---

                # If element smaller than pivot is found
                # swap it with the greater element pointed by i
                i += 1

                # print(f"i = {i}") ---
                # print(f"changing places {arr[i]} and {arr[j]}") ---

                # Swapping element at i with element at j
                arr[i], arr[j] = arr[j], arr[i]

                # print(arr) ---
        # print(f"changing places {arr[i+1]} and {arr[high]}") ---

        # Swap the pivot element with
        # the greater element specified by i
        arr[i + 1], arr[high] = arr[high], arr[i + 1]

        # print(arr) ---

        # Return the position from where partition is done
        return i + 1

    if low < high:
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, low, high)

        # Recursive call on the left of pivot
        quick_sort_good(array, low, pi - 1)

        # Recursive call on the right of pivot
        quick_sort_good(array, pi + 1, high)


# radix sort O(d*(n+b))
# d - the number of digits
# n - number of elements
# b - base of the number system being used
# Radix sort is a non-comparative integer sorting algorithm
# In practical implementations, radix sort is often faster than
# other comparison-based sorting algorithms, such as quicksort or
# merge sort, for large datasets, especially when the keys have many
# digits. However, its time complexity grows linearly with the number
# of digits, and so it is not as efficient for small datasets.
def radix_sort(arr: [float]) -> None:
    print(arr)

    # A function to do counting sort of arr[] according to
    # the digit represented by exp.
    def countingSort(arr: [float], exp1: int) -> None:

        n = len(arr)

        # The output array elements that will have sorted arr
        output = [0] * (n)
        # print(f"output = {output}") ---

        # initialize count array as 0
        count = [0] * (10)
        # print(f"count = {count}") ---

        # print(f"-------") ---

        # Store count of occurrences in count[]
        for i in range(0, n):
            index = arr[i] // exp1
            count[index % 10] += 1
            # print(f"count = {count}") ---

        # print(f"-------") ---

        # Change count[i] so that count[i] now contains actual
        # position of this digit in output array
        for i in range(1, 10):
            count[i] += count[i - 1]
            # print(f"count = {count}") ---

        # print(f"-------") ---

        # Build the output array
        i = n - 1
        while i >= 0:
            index = arr[i] // exp1
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
            # print(f"output = {output}") ---
            # print(f"count = {count}") ---

        # Copying the output array to arr[],
        # so that arr now contains sorted numbers
        i = 0
        for i in range(0, len(arr)):
            arr[i] = output[i]

    # Find the maximum number to know number of digits
    max1 = max(arr)

    # Do counting sort for every digit. Note that instead
    # of passing digit number, exp is passed. exp is 10^i
    # where i is current digit number
    exp = 1
    while max1 / exp >= 1:
        # print(f"exp = {exp}") ---
        countingSort(arr, exp)
        exp *= 10


# bogo sort O(0-inf)
def bogo_sort(arr: [float]) -> None:
    def is_sorted(arr: [float]) -> bool:
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]: return False
        return True

    def Fisher_Yates_shuffle(arr: [float]) -> [float]:
        for i in range(len(arr) - 1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    while not is_sorted(arr):
        print(arr)
        Fisher_Yates_shuffle(arr)


# shell sort (Shell sort is mainly a variation of Insertion Sort.)
# Time complexity of the above implementation of Shell sort is O(n^2).
# In the above implementation, the gap is reduced by half in every iteration.
# There are many other ways to reduce gaps which leads to better time complexity.
# The worst-case complexity for shell sort is  O(n^2)
# The shell sort Average Case Complexity depends on the interval selected by the programmer. O(n log n)
def shell_sort(arr: [float], n: int = None):
    if n == None: n = len(arr)

    gap = n // 2

    while gap > 0:
        j = gap
        # Check the array in from left to right
        # Till the last possible index of j
        while j < n:
            i = j - gap  # This will keep help in maintain gap value

            while i >= 0:
                # If value on right side is already greater than left side value
                # We don't do swap else we swap
                if arr[i + gap] > arr[i]:

                    break
                else:
                    arr[i + gap], arr[i] = arr[i], arr[i + gap]

                i = i - gap  # To check left side also
                # If the element present is greater than current element
            j += 1
        gap = gap // 2


# heap sort O(n log n)
# Typically 2-3 times slower than well-implemented QuickSort.
def heap_sort(arr: [float]) -> None:
    def heapify(arr: [float], N: int, i: int) -> None:
        largest = i  # Initialize largest as root
        l = 2 * i + 1  # left = 2*i + 1
        r = 2 * i + 2  # right = 2*i + 2

        # See if left child of root exists and is
        # greater than root
        if l < N and arr[largest] < arr[l]:
            largest = l

        # See if right child of root exists and is
        # greater than root
        if r < N and arr[largest] < arr[r]:
            largest = r

        # Change root, if needed
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]  # swap

            # Heapify the root.
            heapify(arr, N, largest)

    N = len(arr)

    # Build a maxheap.
    for i in range(N // 2 - 1, -1, -1):
        heapify(arr, N, i)

    # One by one extract elements
    for i in range(N - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)


# stalin sort O(n)
def stalin_sort(arr: [float]) -> None:
    i = 0
    length = len(arr)
    if length > 0:
        aux = arr[0]
    while i < length and length > 0:
        if aux <= arr[i]:
            aux = arr[i]
            i = i + 1
        else:
            del arr[i]
            length = length - 1


arr = [5, 3, 4, 2]

# bubble_sort(arr)
# insertion_sort(arr)
# selection_sort(arr)
# arr = merge_sort(arr)
# arr = quick_sort_bad(arr)
# quick_sort_good(arr)
# radix_sort(arr)
# bogo_sort(arr)
# shell_sort(arr)
# heap_sort(arr)
# stalin_sort(arr)

print(arr)
