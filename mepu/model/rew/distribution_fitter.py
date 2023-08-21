import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

def get_distribution(record_bg, record_kn, picture_name = None):
    
    x = np.linspace(0,5,501)
    params_bg = scipy.stats.exponweib.fit(record_bg)
    y_bg = scipy.stats.exponweib.pdf(x,*params_bg)
    params_kn = scipy.stats.exponweib.fit(record_kn)
    y_kn = scipy.stats.exponweib.pdf(x,*params_kn)

    if picture_name != None:
        bg_mean = np.mean(record_bg)
        kn_mean = np.mean(record_kn)
        for idx, x_val in enumerate(x):
            if x_val > bg_mean and x_val > kn_mean:
                if y_bg[idx] < 0.05 and y_kn[idx] < 0.05:
                    max_x = x_val
                    break
                
        for idx, x_val in enumerate(x):
            if x_val < bg_mean and x_val < kn_mean:
                if y_bg[idx] > 0.05 or y_kn[idx] > 0.05:
                    min_x = x_val
                    break
            
        plt.figure(dpi=300)
        plt.xlim(min_x, max_x)
        plt.plot(x, y_bg, color = "b", label = "weibull model of background")
        plt.plot(x, y_kn, color = "r", label = "weibull model of foreground")
        plt.hist(record_bg, range=[0, max_x], density = True, color = "b", bins = 50, rwidth = 0.85, alpha = 0.75, label="background regions")
        plt.hist(record_kn, range=[0, max_x], density = True, color = "r", bins = 50, rwidth = 0.85, alpha = 0.75, label="known object regions")
        plt.xlabel("Reconstruction Errors")
        plt.ylabel("Normalized Histogram")
        
        plt.legend()
        plt.show()
        if not os.path.exists("vis_dist"):
            os.mkdir("vis_dist")
        plt.savefig(os.path.join("vis_dist", picture_name))
        plt.close()
    return y_bg, y_kn

