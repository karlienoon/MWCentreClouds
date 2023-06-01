import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy import units as u
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap

########################################################################
# Define the Radio maps class
# This is a class that reads and performes different types of  
# analysis on 2D radio maps.
########################################################################
class Radio_maps:
    def __init__(self, filename, dist):
        '''
        Initialises the Radio_maps Class. 
        This module opens and reads FITS files, and defines the map's 
        coordinate system, min and max values, pixel size, and 
        calculates levels and the direction of the Galactic Centre

        Parameters
            filename : string
                the directory link to a radio cube fits file
            dist: float
                distance to the object in kpc
        Returns
            Nothing
        '''
        # Open the FITS file and read the data
        self.filename = filename
        fits_file = fits.open(self.filename)[0]
        self.data, self.header = fits_file.data, fits_file.header
        self.wcs = WCS(self.header)
        self.lon = self.wcs.wcs_pix2world(np.arange(self.data.shape[1]
                                                    ),0,0)[0]
        self.lat = self.wcs.wcs_pix2world(0,np.arange(self.data.shape
                                                      [0]),0)[1]
        self.extent = [self.lon[0],self.lon[-1],self.lat[0],self.lat
                       [-1]]
        self.vmin = np.nanmin(self.data)
        self.vmax = np.nanmax(self.data)
        self.levels = np.linspace(self.vmin, self.vmax, 6)
        self.pix = np.abs(self.header['CDELT1'])
        self.pixcm = dist*np.tan(np.radians(self.pix))*3.086E21
        self.m = (self.lat[-1]-self.lat[0])/(self.lon[-1]-self.lon[0])
        self.initial_pix = int(self.data.shape[0]/2)
        x_step = 16
        y_step = self.m*x_step
        self.dx = self.lon[self.initial_pix]-self.lon[self.initial_pix
                                                      -x_step]
        self.dy = self.lat[self.initial_pix]-self.lat[
            int(self.initial_pix-np.abs(np.rint(y_step)))]
        self.ticks = np.around(self.levels, decimals=1)

    ########################################################################
    # Method to generate an annotated column density map from a 0th moment 
    # map
    ########################################################################
    def column_density_map(self, species):
        '''
        This method generates an annotated column density map including 
        beam size, physical scale, object name and arrow pointing towards
        the GC

        Parameters
            species: string
                e.g. 'HI' or 'CO'
        Returns
            figure: pdf
                Column density map in pdf format
        '''
        # Generating figure
        fig, axs = plt.subplots(1,1,gridspec_kw={"width_ratios":[1], 
                                                 'height_ratios': [1],
                                                 'wspace':0.0}
                                ,figsize=(20,20), constrained_layout=True)
        row_subplot = 0
        ax1 = axs
        # Defining figure variables
        fontsize = 18
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=1)
        props2 = dict(boxstyle='round', facecolor='w', alpha=1, 
                      edgecolor='w')

        mapm0=ax1.imshow(self.data,origin='lower',cmap='BuPu',aspect='auto', 
                         extent=self.extent,vmin=self.vmin, vmax=self.vmax)
        cs = ax1.contour(self.data,colors='k', levels=self.levels, 
                         extent=self.extent)
        cb = fig.colorbar(mapm0, ax = ax1, orientation = 'horizontal', 
                          location="top", ticks=self.ticks)
        cb.ax.tick_params(labelsize=fontsize)
        # Changes label of colorbar depending on species
        if species == 'HI':
            cb.set_label(
                r'$N_\mathrm{HI} \times \mathrm{10}^{20}$ ($\mathrm{cm}^{-2}$)'
                , labelpad=5, fontsize=fontsize)
        else:
             cb.set_label(
                 r'$N_\mathrm{H2} \times \mathrm{10}^{20}$ ($\mathrm{cm}^{-2}$)'
                 , labelpad=5, fontsize=fontsize)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('bottom')
        ax1.grid()
        ax1.set_xlabel(self.header["CTYPE1"][:4]+"("+self.header["CUNIT1"]+")"
                       ,fontsize=fontsize)
        ax1.set_ylabel(self.header["CTYPE2"][:4]+" ("+self.header["CUNIT2"]+")"
                       ,fontsize=fontsize)
        ax1.tick_params(direction='in',labelsize=fontsize, length=10, 
                        width=3)
        ax1.text(.02, .98, self.header["OBJECT"], 
                 fontsize=fontsize,ha='left', va='top', 
                 transform=axs.transAxes, bbox=props)

        # GC arrow
        ax1.arrow(self.lon[int(self.lon.shape[0]*(1/3))], 
                  self.lat[int(self.lat.shape[0]*(3/4))], 
                  -self.dx, self.dy, head_width = 0.01, width = 0.0005)

        # Physical scale
        xmin_scale = int(self.lon.shape[0]*(2/3))
        five_arcmin = int(5/(self.pix*60))
        ax1.text(self.lon[xmin_scale+20], self.lat[10]
                 , r'5$^{\prime}$ $\simeq$ 12 pc', 
                 fontsize=18,ha='right',va='bottom', bbox=props2)
        ax1.hlines(y=self.lat[5], xmin=self.lon[xmin_scale], 
                   xmax=self.lon[xmin_scale+five_arcmin], linewidth=3, 
                   color='k')
        # Beam ellipse
        bmaj = (26.1057202736292*u.arcsec).to(u.deg)
        bmin = (21.5060080646916*u.arcsec).to(u.deg)
        ellipse = Ellipse(xy=(.25+.5, .25), width=bmaj.value , 
                          height=bmin.value, lw=2, angle=0, fill=False, 
                          color='red')
        ax1.add_artist(ellipse)

        plt.savefig(self.filename[:-5]+"CD_map.pdf",bbox_inches='tight')

    ########################################################################
    # Method to generate an overlay map given two data sets in the same
    # coordinate system
    ########################################################################
    def overlays(self, other):
        '''
        This method overlays CO and HI column density maps

        Parameters
            other: class
                CO class
        Returns
            figure: pdf
                Overlayed HI and CO column density maps in pdf format
        '''
        # Generating the figure
        fig, axs = plt.subplots(1,1,gridspec_kw={"width_ratios":[1.]},
                                figsize=(20,20),constrained_layout=True)
        row_subplot = 0
        ax = axs
        # Figure parameters
        fontsize = 18
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=1)
        # Plotting HI
        image_field = ax.imshow(self.data, origin='lower', aspect='auto'
                                ,cmap='Greys',  extent=self.extent, 
                                interpolation='bilinear', rasterized=True
                                , vmin=self.vmin, vmax=self.vmax)
        ax.contour(self.data, origin='lower', levels=self.levels, 
                   colors='k', linewidths=0.7, alpha=1, 
                   linestyles='solid', extent=self.extent)
        # Plotting CO
        ax.contourf(other.data, levels=other.levels, cmap='BuPu', 
                    vmin=other.vmin, vmax=other.vmax, 
                    extent=other.extent)
        ax.contour(other.data, levels=other.levels, colors='k', 
                   linewidths=0.7, 
                   linestyles='solid', extent=other.extent)
        # Below line is useful if CO map is prefered over contourf
        #image_density = ax.imshow(other.data, origin='lower', 
                                # aspect='auto',cmap='BuPu', alpha=1, 
                                # vmin=other.vmin, vmax=other.vmax, 
                                # rasterized=True)
        
        # The below 4 lines plot the footprint of C1 CO observations
        #ax.plot((lon1[86], lon1[0]), (lat1[0],lat1[49]), c='k', linestyle=':')
        #ax.plot((lon1[0], lon1[49]), (lat1[50],lat1[136]), c='k', linestyle=':')
        #ax.plot((lon1[87], lon1[136]), (lat1[0],lat1[86]), c='k', linestyle=':')
        #ax.plot((lon1[136], lon1[50]), (lat1[87],lat1[136]), c='k', linestyle=':')

        ax.tick_params(direction='in',labelsize=fontsize, 
                       length=10, width=3)
        ax.set_xlabel(self.header["CTYPE1"][:4]+" ("+self.header["CUNIT1"]+")"
                      , fontsize=fontsize)
        ax.set_ylabel(self.header["CTYPE2"][:4]+" ("+self.header["CUNIT2"]+")"
                      , fontsize=fontsize)
        # Annotations
        ax.text(.02, .98, self.header["OBJECT"], fontsize=fontsize,ha='left'
                , va='top', transform=axs.transAxes, bbox=props)
        ax.arrow(self.lon[int(self.lon.shape[0]*(1/3))], 
                 self.lat[int(self.lat.shape[0]*(3/4))],
                -self.dx, self.dy, head_width = 0.008, 
                width = 0.0005)
        # Save figure
        plt.savefig(self.filename[:-5]+"overlay_map.pdf",bbox_inches='tight')

    ########################################################################
    # Method to calculate the mass in units solar masses from a column 
    # density map
    ########################################################################
    def mass(self, Hnuclei):
        '''
        Calculates the mass in units solar mass of a column density map in H nuclei/cm^2.

        Parameters:
            Hnuclei: integer
                # of H nuclei in species e.g. 1 or 2

        Returns:
            Mass: string
                Mass to 3 significant figures in units solar mass
        '''
        # HI mass map in units of kg, 1.36 factor to account for helium,
        # 1.67e-27 factor is the mass of a hydrogen atom, Hnuclei is a 
        # parameter to account for elements other than HI. Hnuclei is 1 if 
        # the data is HI, 2 if it is CO. 
        MHImap1 =(1.6726219E-27*Hnuclei*self.data*(self.pixcm)**2)*1.36
        # HI mass map in units of Msun
        MHImap1 /= 2.0E30
        # Total HI mass
        HImass1 = np.nansum(MHImap1)
        # Print results
        print ('Mass: %.3f Msun'%(HImass1))

    ########################################################################
    # Method generates two arrays containing all detections when there is a 
    # non-nan value in both HI and CO column density maps
    ########################################################################
    def detections(self, other):
        '''
        Gives two arrays containing the column densities of points that are 
        detected in both CO and HI.

        Parameters:
            other: class
                CO class

        Returns:
            HI: array
                Array containing all HI detections where there is also
                a CO detection
            CO: array
                Array containing all CO detections where there is also
                a HI detection
        '''
        a, b = self.data.flatten(), other.data.flatten()    
        detections = []
        for i in range(len(b)):
            if np.isnan(b[i]) == False:
                if np.isnan(a[i]) == False:
                    detections.append(i)
                
        CO_common = []
        for k in range(len(detections)):
            CO_common.append(b[detections[k]])
                
        HI_common = []
        for j in range(len(detections)):
            HI_common.append(a[detections[j]])
    
        return np.array(HI_common), np.array(CO_common)
    
    ########################################################################
    # Method generates a histogram containing all HI detections vs only 
    # detections in both CO and HI
    ########################################################################
    def hist(self, other):
        '''
        Generates a histogram of total HI detections and 
        common HI and CO detections

        Parameters:
            other: class
                CO class

        Returns:
            figure: pdf
                Saves histogram in pdf format
        '''
        # Defining parameters
        fontsize = 18
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=1)
        CO_c = 'darkmagenta'
        HI_c = 'lightsteelblue'

        # Calculation common detections
        NHI, NH2 = self.detections(other)

        # Generating new array based on original HI data with no nans
        flatdetec = []
        for i in range(len(self.data.flatten())):
            if np.isnan(self.data.flatten()[i]) == False:
                flatdetec.append(self.data.flatten()[i])
        
        # Generating figure
        fig, axs = plt.subplots(1,1,gridspec_kw={"width_ratios":[1],
                                                 'wspace': 0.1},
                                                 figsize=(20,20),
                                constrained_layout=True, sharex=True)
        row_subplot = 0
        ax = axs
        
        # Plotting total HI detections
        ax.hist(np.log10(np.array(flatdetec)), density=False, bins=30, 
                color='k', label='HI total', alpha=0.5, rasterized=True) 
        # Plotting common HI detections
        ax.hist(np.log10(NHI), density=False, bins=30, color=HI_c, 
                label='HI ($\exists$ H$_2$)', alpha=1, rasterized=True)
        # Plotting common CO detections 
        ax.hist(np.log10(NH2), density=False, bins=30, color=CO_c, 
                label='H$_2$ ($\exists$ HI)', alpha=0.7, rasterized=True)
        # Formatting and annotations
        ax.legend(fontsize=16)
        ax.text(.02, .98, self.header["OBJECT"], fontsize=fontsize,
                ha='left', va='top', transform=axs.transAxes, bbox=props)
        ax.tick_params(which="minor", axis="x", direction="in", 
                       pad=15, length=6)
        ax.tick_params(which="minor", axis="y", direction="in", 
                       length=6)
        ax.tick_params(which="major", axis="x", direction="in", 
                       pad=15, length=10)
        ax.tick_params(which="major", axis="y", direction="in", 
                       length=10)
        # Saving figure
        plt.savefig(self.filename[:-5]+"_histogram.pdf",bbox_inches='tight')

    ########################################################################
    # Method generates two arrays containing all detections when there is a 
    # non-nan value in both HI and CO column density maps. If there is a 
    # nan value in CO, the method generates an upper limit for CO using 
    # the CO noisemap and masking level.
    ########################################################################
    def detections_limit(self, other, noisemap, mask_level):
        '''
        Gives two arrays containing the column densities of points that are 
        detected in HI and CO. If there is a detection in HI but not in CO,
        an upper limit is calculated for CO using the CO noisemap and the 
        masking level

        Parameters:
            other: class
                CO class
            noisemap: array
                CO noisemap fits file
            mask_level: integer
                Level of mask applied to CO data

        Returns:
            HI: array
                Array containing all HI detections
            CO: array
                Array containing all CO detections and proxy detections
                where there is a HI detection
        '''
        a, b, noisemap = self.data.flatten(), other.data.flatten(), noisemap.data.flatten()
        detections = []
        # finding detections in HI but no CO
        for i in range(len(b)):
            if np.isnan(a[i]) == False:
                if np.isnan(b[i]) == True:
                    if np.isnan(noisemap[i]) == False:
                        detections.append(i)
        # Generating proxy, CO detections (upper limit)
        CO_common = []
        for k in range(len(detections)):
            CO_common.append(noisemap[detections[k]]*mask_level)
        # Listing HI detections        
        HI_common = []
        for j in range(len(detections)):
            HI_common.append(a[detections[j]])
    
        return np.array(HI_common), np.array(CO_common)

    ########################################################################
    # Method produces a surface density map (Etot vs EHI)
    ########################################################################
    def SD_scatter(self, other, noisemap, mask_level):
        '''
        Produces a surface density map in units solar mass per squared pc

        Parameters:
            other: class
                CO class
            noisemap: array
                CO noisemap fits file
            mask_level: integer
                Level of mask applied to CO data

        Returns:
            figure: pdf
                Figure of surface density map
        '''
        # Converts column densities into grams
        def attog(value):
            atomtograms = ((value*2.2712e-24)*3.09e18**2)/(1.99e33)
            return atomtograms
        # Calculate common detections
        NHI, NH2 = self.detections(other)
        # Turn column densities into grams
        NHI, NH2 = attog(NHI), attog(NH2)
        # Calculating Ntotal
        Ntot = NHI + NH2
        # Calculating CO upper limits
        NHI_L, NH2_L = self.detections_limit(other, noisemap, mask_level)
        NHI_L, NH2_L = attog(NHI_L), attog(NH2_L)
        # Calculating mean and max of upper limits
        NHI_max = NHI_L +  NH2_L
        NHI_mean = 0.5 * (NHI_L + NHI_max)

        # Generating figure
        fig, axs = plt.subplots(1,1,figsize=(15,10),constrained_layout=True)
        # Plotting parameters
        xh2 = np.linspace(0.1, 100., 200)
        xx = np.linspace(0.1,8.8, 10)
        x_lim = 150
        ds = 140 
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=1)

        # percentile lines
        first = 0.3
        second = 0.1
        third = 0.03
        model_c = 'k'
        low = 'red'
        midd = 'red'
        high = 'red'
        ls = (0, (5, 10))
        ls2 = (0, (5, 5))
        ls3 = (0, (5, 1))

        row_subplot = 0
        ax = axs

        # Plotting Ntotal vs NHI
        ax.scatter(Ntot,NHI, c='rebeccapurple', label='This work', s=120,
                    marker='^', alpha=0.6)
        # Adding upper limits for CO
        ax.errorbar(NHI_mean, NHI_L, xerr=[NHI_mean-NHI_L, NHI_max-NHI_mean]
                    , elinewidth=0.1, fmt="none", c='grey', alpha=0.04)

        # Including theoretical line
        ax.plot(xx, xx, c=model_c)
        ax.hlines(y=8.8, xmin=8.8, xmax=x_lim, color=model_c)
        ax.text(0.2,0.2,'KMT model', rotation=45, va='baseline',ha='center'
                , fontsize='large',c=model_c)
        # This theoretical line corresponds to Z = solar Z 
        ax.text(72,10,'Z=Z$_\odot$', rotation=0, va='baseline',ha='center'
                , fontsize='large',c=model_c)

        # Plotting multiple lines for different Z values
        ax.plot(2*xx, 2*xx, c=model_c)
        ax.hlines(y=2*8.8, xmin=2*8.8, xmax=x_lim, color=model_c)
        ax.text(74,20,'Z=0.5Z$_\odot$', rotation=0, va='baseline',ha='center'
                , fontsize='large',c=model_c)
        ax.plot(0.5*xx, 0.5*xx, c=model_c)
        ax.hlines(y=0.5*8.8, xmin=0.5*8.8, xmax=x_lim, color=model_c)
        ax.text(72,5,'Z=2Z$_\odot$', rotation=0, va='baseline',ha='center'
                , fontsize='large',c=model_c)

        # Percentile lines
        # 70%
        ax.plot(xh2,xh2*first, c=high, ls=ls3)
        ax.text(0.38, 0.11,'f$_{\mathrm{H}_{\mathrm{2}}}=$70%', rotation=45,
                 va='baseline',ha='center', fontsize='large',c=high)
        # 90%
        ax.plot(xh2,xh2*second, c=midd, ls=ls2)
        ax.text(xh2[2],xh2[2]*second,'f$_{\mathrm{H}_{\mathrm{2}}}=$90%',
                 rotation=45, va='baseline',ha='center', fontsize='large',
                 c=midd)
        # 97%
        ax.plot(xh2,xh2*third, c=low, ls=ls)
        ax.text(xh2[7],xh2[7]*third,'f$_{\mathrm{H}_{\mathrm{2}}}=$97%',
                 rotation=45, va='baseline',ha='center', fontsize='large',
                 c=low)

        # Plot formatting
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([0.1, 1., 10])
        ax.set_xticks([0.1, 1., 10, 100])
        ax.tick_params(which="minor", axis="x", direction="in", 
                       pad=15, length=6)
        ax.tick_params(which="minor", axis="y", direction="in", length=6)
        ax.tick_params(which="major", axis="x", direction="in", pad=15, 
                       length=10)
        ax.tick_params(which="major", axis="y", direction="in", length=10)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_ylim(0.1, 30.)
        ax.set_xlim(0.1, 100.)
        ax.tick_params(labelsize=18)
        # Axis labels
        fig.text(0.5, -0.03, 
                 '$\Sigma_{\mathrm{at}}+\Sigma_{\mathrm{mol}} (\mathrm{M}_{\mathrm{\odot}} \mathrm{pc^{-2}})$'
                 , ha='center', fontsize=18)
        fig.text(-0.05, 0.5, 
                 '$\Sigma_{\mathrm{at}}(\mathrm{M}_{\mathrm{\odot}}  \mathrm{pc^{-2}})$'
                 , va='center', rotation='vertical', fontsize=18)
        # Object labels
        ax.text(75,0.14, self.header["OBJECT"], fontsize=20,
                verticalalignment='top', bbox=props)
        # Legend formatting
        leg = ax.legend(loc='upper left', fontsize='18',markerscale=1.5)
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        # Save figure
        plt.savefig(self.filename[:-5]+"_surface_density.pdf",
                    bbox_inches='tight')
        
    ########################################################################
    # Method produces an array for CO that uses upper limits and an array 
    # HI that replaces nan values with 0
    ########################################################################
    def mole_frac_upperL(self, other, noisemap, mask_level):
        '''
        Produces two arrays: 
        One using upper limits for CO values when there is a positive HI 
        detections but a nan CO. The other array gives the HI dectections
        when there is a positive CO detection whilst replacing nan values
        with zero.

        Parameters:
            other: class
                CO class
            noisemap: array
                CO noisemap fits file
            mask_level: integer
                Level of mask applied to CO data

        Returns:
            figure: pdf
                Figure of two molecular fraction maps
        '''
        a, b = self.data, other.data
        # Calculating upper limits for CO
        for l in range(len(b)):
            for k in range(len(b)):
                if np.isnan(a[l,k]) == False:
                    if np.isnan(b[l,k]) == True:
                        b[l,k] = noisemap.data[l,k]*mask_level
    
        # Setting a nan detection in HI as 0 
        for j in range(len(b)):
            for i in range(len(b)):
                if np.isnan(b[j,i]) == False:
                    if np.isnan(a[j,i]) == True:
                        a[j,i] = 0.0
    
        return a,b

    ########################################################################
    # Method produces two molecular fraction maps
    ########################################################################
    def mol_frac_maps(self, other, noisemap, mask_level):
        '''
        Produces two molecular fraction maps: 
        One showing the molecular fractions when there are common detections
        and one using an upper limit for the CO

        Parameters:
            other: class
                CO class
            noisemap: array
                CO noisemap fits file
            mask_level: integer
                Level of mask applied to CO data

        Returns:
            figure: pdf
                Figure of two molecular fraction maps
        '''
        # Generating figure
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,10),
                                constrained_layout=True)
        # Defining figure parameters
        interp = 'bicubic'
        levels=[4.625e+20, 7.75e20,10.875e20, 14e20]
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=1)
        # Generate colormap colors, multiply them with the factor "a", 
        # and create new colormap
        a = 0.9
        my_cmap = plt.cm.BuPu(np.arange(plt.cm.BuPu.N))
        my_cmap[:,0:3] *= a 
        cmaps = ListedColormap(my_cmap)

        # Generating common detections
        detections = other.data/(self.data+other.data)
        NHI, NH2 = self.mole_frac_upperL(other, noisemap, mask_level)
        # Calculating upper limit data
        detections_L = NH2/(NHI+NH2)

        # Plotting common detections
        axs[0].imshow(detections, origin='lower',
                  aspect='auto', cmap=cmaps, interpolation=interp,vmin=0
                  , vmax=1, extent=self.extent)
        axs[0].contour(other.data, levels=levels, colors='w', 
                       linewidths=1.2, extent=self.extent)
        axs[0].contourf(detections, levels=4, cmap='BuPu', 
                        extent=self.extent)
        # Plotting C1 CO footprint
        axs[0].plot((self.lon[86], self.lon[0]), 
                    (self.lat[0],self.lat[49]), c='k', linestyle=':')
        axs[0].plot((self.lon[0], self.lon[49]), 
                    (self.lat[50],self.lat[136]), c='k', linestyle=':')
        axs[0].plot((self.lon[87], self.lon[136]), 
                    (self.lat[0],self.lat[86]), c='k', linestyle=':')
        axs[0].plot((self.lon[136], self.lon[50]), 
                    (self.lat[87],self.lat[136]), c='k', linestyle=':')
        
        # Figure formatting
        axs[0].tick_params(which="minor", axis="x", 
                           direction="in", pad=15, length=6)
        axs[0].tick_params(which="minor", axis="y", 
                           direction="in", length=6)
        axs[0].tick_params(which="major", axis="x", 
                           direction="in", pad=15, length=10)
        axs[0].tick_params(which="major", axis="y", 
                           direction="in", length=10)
        axs[0].text(self.lon[5],self.lat[130], self.header["OBJECT"]
                    , fontsize=20,verticalalignment='top', bbox=props)
        axs[0].tick_params(labelsize=18)
        axs[0].set_ylabel(
            self.header["CTYPE2"][:4]+" ("+self.header["CUNIT2"]+")"
            , fontsize=18)

        # Second map including CO upper limits
        axs[1].imshow(detections_L, origin='lower',
                  aspect='auto', cmap=cmaps,interpolation=interp
                  , vmin=0, vmax=1, extent=self.extent)
        axs[1].contourf(detections_L, levels=4, cmap='BuPu'
                        , extent=self.extent)
        # Plotting CO contours
        axs[1].contour(other.data, levels=levels, colors='w'
                       , linewidths=1.2, extent=self.extent)
        # CO footprint
        axs[1].plot((self.lon[86], self.lon[0]), 
                    (self.lat[0],self.lat[49]), c='k', linestyle=':')
        axs[1].plot((self.lon[0], self.lon[49]),
                     (self.lat[50],self.lat[136]), c='k', linestyle=':')
        axs[1].plot((self.lon[87], self.lon[136]), 
                    (self.lat[0],self.lat[86]), c='k', linestyle=':')
        axs[1].plot((self.lon[136], self.lon[50]), 
                    (self.lat[87],self.lat[136]), c='k', linestyle=':')
        # Plot formatting
        axs[1].tick_params(which="minor", axis="x", 
                           direction="in", pad=15, length=6)
        axs[1].tick_params(which="minor", axis="y", 
                           direction="in", length=6)
        axs[1].tick_params(which="major", axis="x", 
                           direction="in", pad=15, length=10)
        axs[1].tick_params(which="major", axis="y", 
                           direction="in", length=10)
        axs[1].tick_params(labelsize=18)
        fig.supxlabel( 
            self.header["CTYPE1"][:4]+" ("+self.header["CUNIT1"]+")",
              ha='center', fontsize=18)
        # Save figure
        plt.savefig(self.filename[:-5]+"_mol_frac.pdf",bbox_inches='tight')
    

