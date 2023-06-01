import numpy as np
from astropy.io import fits
from spectral_cube import SpectralCube
from astropy import units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import binned_statistic_2d


########################################################################
# Define the Radio cube utilities class
# This is a class that reads and performes different types of  
# analysis on 3D radio data cubes.
########################################################################
class Radio_cube_utilities:
    def __init__(self, filename):
        '''
        Initialises the Radio_cube_utilities Class. 
        This module opens and reads FITS files, defines coordinate 
        system, velocity and channel width of the cube

        Parameters
            filename : string
                the directory link to a radio cube fits file
        Returns
            Nothing
        '''
        self.filename = filename
        fits_file = fits.open(filename)[0]
        self.data, self.header = fits_file.data, fits_file.header
        self.cube = SpectralCube.read(fits_file)
        self.vel = self.cube.spectral_axis.value
        self.wcs = WCS(self.header)
        self.lon = self.wcs.wcs_pix2world(np.arange(self.data.shape[2])
                                          ,0,0,0)[0]
        self.lat = self.wcs.wcs_pix2world(0,np.arange(self.data.shape[1])
                                          ,0,0)[1]
        self.extent = [self.lon[0],self.lon[-1],self.lat[0],self.lat[-1]]
        self.chw = np.abs(self.header['CDELT3'])
    
    ########################################################################
    # Method to convert radio cubes to K km/s
    ########################################################################
    def data_units(self):
        '''
        Convert radio cubes in units Jy/beam Hz to K km/s.

        Returns:
            cube: array
                Data cube in units K km/s
        '''
        #Allow large operations
        self.cube.allow_huge_operations=True
        #Convert to km/s
        cube = self.cube.with_spectral_unit(u.km/u.s, 
                                            velocity_convention='radio')
        #Convert from Jy/beam to Kelvin
        cube = cube.to(u.K)
        cube.header['BUNIT'] = 'K'
        # Write the cube to a new FITS file
        cube.write(self.filename[:-5]+"_K_km_s.fits", overwrite='True')

    ########################################################################
    # Method to compute moment maps
    ########################################################################
    def moment_maps(self, order, species):
        '''
        This module calculates moment maps of PPV data cubes. If order = 0, 
        the column density map of the specified species will be calculated 
        in units H nuclei/cm^2.

        Parameters:
            order: integer
                Number of moment map
            species: string
                Name of species e.g. 'HI' or 'CO'

        Returns:
            cube: array
                Data cube in units K km/s
        '''
        #Calculate the moment map of choosen order. Note, to produce 
        # accurate second moment maps, the cube must be well masked 
        # spatially and spectrally
        cube_moment = self.cube.moment(order=order)
        # Calculate column density map
        if species == 'HI':
            if order == 0:
                cube_moment = cube_moment*1.823E18
        if species == 'CO':
            if order == 0:
                cube_moment = cube_moment*4E20
        # Write the moment map to a new FITS file
        hdu = fits.PrimaryHDU(cube_moment.value, header=cube_moment.header)
        hdu.writeto(self.filename[:-5]+"_m%i.fits"%(order), 
                    overwrite='True')

    ########################################################################
    # Method to mask radio cubes
    ########################################################################
    def mask_HI(self, sigma, Nchn):
        '''
        This module masks a HI PPV cube in units K km/s to produce a masked 
        column density map in units H nuclei/cm^2.

        Parameters:
            sigma: integer  
                Mask level
            Nchn: integer
                Number of emission-free channels used to calculate rms noise

        Returns:
            masked_m0: array
                Saves a masked zeroth moment map in units H nuclei/cm^2
        '''
        # Calculate rms noise of a channel from a selection of 
        # emission-free channels.
        # np.nan_to_num replaces nan values with zeroes
        rms_chn = np.std(np.nan_to_num(self.data[0:Nchn]))
        # Calculate the rms noise of the integrated map
        # 1.2 is to account for dependent channels
        rms = rms_chn * self.chw * np.sqrt(Nchn/1.2)
        # Calculate zeroth moment map
        summap = self.cube.moment(order=0)
        # Mask zeroth moment map
        summap.value[summap.value<sigma*rms] = np.nan
        #summap.header['OBJECT'] = self.cube.header['OBJECT']
        # Write the masked moment map to a new FITS file
        hdu = fits.PrimaryHDU(1.823E18*summap.value, header=summap.header)
        hdu.writeto(self.filename[:-5]+"_masked%i_m0.fits"%(sigma), 
                    overwrite='True')

    ########################################################################
    # Method to mask CO cubes
    ########################################################################
    def mask_CO(self, sigma, Nchn):    
        '''
        This module masks a CO PPV cube to a specified mask level. 
        Calculates column density in units H nuclei/cm^2.

        Parameters:
            sigma: integer  
                Mask level
            Nchn: integer
                Number of emission-free channels used at the start and end
                of a data cube to calculate rms noise

        Returns:
            masked_m0: array
                Saves a masked zeroth moment map in units H nuclei/cm^2
            noisemap: array
                Saves a noisemap generated from the unmasked data cube
        '''
        # Creates an emission-free data cube using the first and last 20 
        # channels
        emiss_free_cube = (self.data)[np.r_[0:Nchn, self.data.shape[0]
                                            -Nchn:self.data.shape[0]]]
        # Defining channel width
        #Calculating std across the emissionless cubes
        raw_std = np.std(emiss_free_cube, axis=0)*self.chw*np.sqrt(self.data.shape[0])
        # Calculate zeroth moment map
        summap = self.cube.moment(order=0)
        #Constructing noise map 
        noise_map_arr = np.array(raw_std).reshape(summap.shape[1], 
                                                  summap.shape[1])
        #Constructing S/N map
        sn_cube = summap.value/noise_map_arr
        # Turning the S/N cube into the masking cube
        sn_cube[sn_cube<sigma] = np.nan
        #Masking moment map
        m = np.ma.masked_where(np.isnan(sn_cube), summap.value)
        M = np.ma.filled(m, np.nan)
        summap.header['OBJECT'] = self.cube.header['OBJECT']
        # Write the noisemap to a new FITS file
        hdu = fits.PrimaryHDU(noise_map_arr, header=summap.header)
        hdu.writeto(self.filename[:-5]+"_noisemap.fits", overwrite='True') 
        # Write the masked moment map to a new FITS file
        hdu2 = fits.PrimaryHDU(4E20*M, header=summap.header)
        hdu2.writeto(self.filename[:-5]+"_masked%i_m0.fits"%(sigma), 
                     overwrite='True') 

    ########################################################################
    # Method to compute position-velocity maps
    ########################################################################
    def PV_plot(self, dist, other):
        '''
        This module computes the projected distance of a pixel and produces 
        a position-velocity plot of HI and CO on the same figure

        Parameters:
            dist: float
                Distance to the object in units kpc
            other: class
                CO data
        Returns:
            PV plot: figure
                Saves a position-velocity plot in PDF format including 
                both HI and CO
        '''
        # Defining parameters for the figure
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=1)
        my_cmap = plt.cm.BuPu(np.arange(plt.cm.BuPu.N))
        my_cmap[:,0:3] *= 0.95
        cmaps = ListedColormap(my_cmap)
        fontsize = 18
        # Module to calculate projected distance of a pixel from the 
        # Galactic Centre
        def GC_dist(lon,lat, dist):
            ### setting up x, y, z in spherical coordinates from Earth
            pixel_x = dist*np.cos(np.radians(lat))*np.cos(
                np.radians(lon))
            pixel_y = dist*np.cos(np.radians(lat))*np.sin(
                np.radians(lon))
            pixel_z = dist*np.sin(np.radians(lat))
            ### Calculating distance where GC(R, 0, 0) (from Earth) and 
            # d=sqrt((x1-xo)^2+(y1-yo)^2+(z1-zo)^2)
            dist2 = (pixel_x - dist)**2 + pixel_y**2 + pixel_z**2
            GC_dist = np.sqrt(np.abs(dist2))
            return GC_dist
        # Generating a list that includes the projected distance, 
        # velocity and brightness temperature of a pixel
        dist_all1 = []
        for k in range(self.data.shape[0]):
            for j in range(len(self.lat)):
                for i in range(len(self.lon)):
                    if np.isnan(self.data[k][j][i]) == False:
                        dist_all1.append([GC_dist(self.lon[i], 
                                                  self.lat[j], dist), 
                                                  self.vel[k], 
                                                  self.data[k][j][i]])
        # Calculating the 2D statistics of a pixel's projected 
        # distance and velocity weighted by temperature brightness
        sv1 = np.array(dist_all1)
        s1, v1, cd1 = sv1[:,0], sv1[:,1], sv1[:,2]
        # Replacing negative temperture brightness readings with 0
        ib1 = [0 if i < 0 else i for i in cd1]
        statistic1, xedges1, yedges1, binnumber1 = binned_statistic_2d(
            s1, v1, values=ib1, statistic='sum',bins=[100,self.vel.shape[0]])
        # Normalising the intensity
        normed1 = (statistic1.T - np.min(statistic1.T)) / (np.max(statistic1.T) - 
                                                           np.min(statistic1.T))
        # Calculating the same as above but for the CO (other)
        dist_all2 = []
        for k in range(other.data.shape[0]):
            for j in range(len(other.lat)):
                for i in range(len(other.lon)):
                    if np.isnan(other.data[k][j][i]) == False:
                        dist_all2.append([GC_dist(other.lon[i], other.lat[j], dist), 
                                          other.vel[k], other.data[k][j][i]])

        sv2 = np.array(dist_all2)
        s2, v2, cd2 = sv2[:,0], sv2[:,1], sv2[:,2]
        ib2 = [0 if i < 0 else i for i in cd2]
        statistic2, xedges2, yedges2, binnumber2 = binned_statistic_2d(
            s2, v2, values=ib2, statistic='sum',bins=[100,other.vel.shape[0]])
        normed2 = (statistic2.T - np.min(statistic2.T)) / (np.max(statistic2.T) - 
                                                           np.min(statistic2.T))
        # Initialising figure
        fig, axs = plt.subplots(1,1,gridspec_kw={"width_ratios":[1],'hspace': 0.1},
                                figsize=(10,7),constrained_layout=True)
        # Defining levels for HI and CO
        levels1=np.array([0.15, 0.35,0.55, 0.75,  0.95])
        levels2=np.array([0.2, 0.5, 0.8, 1.])
        row_subplot = 0
        ax = axs
        # Defining extent from statistics
        extent1=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
        extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]

        mapp = ax.imshow(normed1, origin='lower',extent=extent1,aspect='auto', 
                         interpolation='bilinear',rasterized=True, 
                         cmap='Greys', vmin=0, vmax=1, alpha=0.7)
        # The below can be included if a CO map is prefered over contourf
        #mapg = ax.imshow(normed2, origin='lower',extent=extent2,aspect='auto', 
        # interpolation='bilinear', cmap=cmaps, vmin=0, vmax=1, alpha=1)
        CS1 = ax.contour(normed1, origin='lower',colors='k',levels=levels1,
                         extent=extent1, linewidths=0.6)
        CS12 = ax.contourf(normed2, origin='lower',cmap=cmaps,levels=levels2,
                           extent=extent2, vmin=0, vmax=1)
        CS122 = ax.contour(normed2, origin='lower',colors='k',levels=levels2,
                           extent=extent2, linewidths=0.6)

        # Making ticks inward and adjusting size
        ax.tick_params(which="minor", axis="x", direction="in", pad=15, length=6)
        ax.tick_params(which="minor", axis="y", direction="in", length=6)
        ax.tick_params(which="major", axis="x", direction="in", pad=15, length=10)
        ax.tick_params(which="major", axis="y", direction="in", length=10)
        
        # Colorbars for both HI map and CO contourf. CS12 should be changed 
        # to mapg if map is preferred
        fig.colorbar(CS12, ax=ax,anchor=(2.0,0.0),aspect=10, 
                     ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        fig.colorbar(mapp, ax=ax, ticks=[],anchor=(1.0,0.0),aspect=10)
        # Object and axis labels
        ax.text(.02, .98, self.header["OBJECT"], fontsize=fontsize,ha='left', 
                va='top', transform=axs.transAxes, bbox=props)
        ax.set_xlabel(self.header["CTYPE1"][:4]+" ("+self.header["CUNIT1"]+")", 
                      fontsize=fontsize)
        ax.set_ylabel(self.header["CTYPE2"][:4]+" ("+self.header["CUNIT2"]+")", 
                      fontsize=fontsize)
        # Save figure
        plt.savefig(self.filename[:-5]+"_PPV.pdf", bbox_inches='tight')

 
        