from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

# return the index in a sorted that is nearest to a specified value;
# this function assumes that the search value is somewhere between
# two of the values inside the array and will return 'None' if not
def nearest_index( sorted_array: np.ndarray, search_value: float ) -> int:
    for index in range(len(sorted_array) - 1):
        if sorted_array[index] < search_value and sorted_array[index + 1] > search_value:
            if search_value - sorted_array[index] < sorted_array[index + 1] - search_value:
                return index
            else: return index + 1    
    return None



# converts a row x column x band rater file into a (row * column) x band data matrix
# (each row in the resultant matrix represents the spectral values of one pixel)
def multispectral_raster_to_matrix( raster: np.ndarray ) -> np.ndarray:
    
    if len(raster.shape) != 3: raise ValueError('Input size expected to be rows x columns x bands.')
    
    rows = raster.shape[0]
    cols = raster.shape[1]
    bands = raster.shape[2]
    
    mat = raster.copy()
    return mat.reshape((rows * cols, bands))



# shortcut function for generating approximate RGB image from multispectral data
# assumes input wavelength array is sorted and in nanometers
def multispectral_raster_to_rgb( raster: np.ndarray, sorted_wavelengths: np.ndarray ) -> np.ndarray:
    
    if len(raster.shape) != 3: raise ValueError('Input size expected to be rows x columns x bands.')
    if len(sorted_wavelengths.shape) != 1: raise ValueError('Input wavelength array expected to be one-dimensional.')
    
    r = nearest_index(sorted_wavelengths, 685)
    g = nearest_index(sorted_wavelengths, 535)
    b = nearest_index(sorted_wavelengths, 475)

    if None in [r,g,b]: raise ValueError('Input wavelength array does not cover expected range of visible spectrum.')

    return raster[:,:,[r,g,b]]

def get_spectrum( data_cube: np.ndarray, pixel: Tuple[int,int] ) -> np.ndarray:
    if len(data_cube.shape) != 3:
        raise ValueError('HSI data cube expected to have 3 dimensions.')
    return data_cube[pixel[0], pixel[1], :]


def spectra_plot( ax: plt.Axes, wavelengths: np.ndarray, spectra: np.ndarray, xlabel='Wavelength (nm)', ylabel = 'Radiance', legend_labels: List[str] = []):
    
    if len(wavelengths) != spectra.shape[1]:
        raise ValueError('Cannot plot spectra; number of wavelengths (x-axis) and samples (y-axis) is not the same.')
    
    if len(legend_labels) > 0 and len(legend_labels) != spectra.shape[0]:
        raise ValueError('Provided list of labels does not correspond to number of spectra to be plotted.')
    
    for i in range(spectra.shape[0]):
        ax.plot(wavelengths, spectra[i,:])

    ax.grid(True)
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(0, 1.10 * np.max(spectra))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(legend_labels) > 0:
        ax.legend(legend_labels)