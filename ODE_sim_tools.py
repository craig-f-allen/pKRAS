import numpy as np
from scipy.integrate import solve_ivp #type: ignore
from scipy.optimize import fsolve #type: ignore
from scipy.optimize import least_squares #type: ignore
import matplotlib.pyplot as plt
from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast
import copy
from multiprocessing import Pool 
import time
from os import cpu_count
from labellines import labelLine, labelLines
from tqdm import tqdm
import pandas as pd

class ODE_Simulation:

    def __init__(self,model_fun,get_param_fun,param_opts):
        self.model_fun = model_fun
        self.get_param_fun = get_param_fun
        self.param_opts = param_opts
        self.k,self.y0 = self.get_param_fun(param_opts)
        self.t = []
        self.y = np.array([])
        self.saved_Sis = []
        self.labels = []
    
    def func(self,x,k):
        WT_RAS_tot = x[0]+x[1]+x[2]+x[4]-k[30]
        Mut_RAS_tot = x[5]+x[6]+x[7]+x[8]-k[31]
        Eff_tot = x[3]+x[4]+x[8]-k[32]
        res = self.model_fun(0,x,k=k)
        res[4:] = np.multiply(res[4:],100)
        #res.append(WT_RAS_tot)
        res.append(Mut_RAS_tot)
        #res.append(Eff_tot)
        return res

    def get_modified_params(self,params,modifications):
        new_param_opts = copy.deepcopy(self.param_opts)
        for i in range(len(params)):
            param = params[i]
            if param['type'] == 'iv':
                new_param_opts[param['name']] = new_param_opts[param['name']]*modifications[i]
        k_new,y0_new = self.get_param_fun(new_param_opts)
        for i in range(len(params)):
            param = params[i]
            if param['type'] == 'kinetic':
                k_new[param['ID']] = k_new[param['ID']]*modifications[i]

        """
        if self.param_opts['mutant'].type == 'dep_kassT':
            k_new[23] = k_new[13]*k_new[16]*((k_new[22]*k_new[21])/(k_new[20]*k_new[14]))/k_new[15]

        elif self.param_opts['mutant'].type == 'dep_kT':
            Haldaneint=(k_new[22]*k_new[21])/(k_new[20]*k_new[23])
            k_new[14] = k_new[13]*k_new[16]*Haldaneint/k_new[15] # new kT
        """
        
        return k_new,y0_new

    def integrate_model(self,t_end,y0=None,k=None,plot_option=False):
        if k is None:
            k = copy.deepcopy(self.k)
        if y0 is None:
            y0 = copy.deepcopy(self.y0)
        sol = solve_ivp(self.model_fun, [0,t_end], y0, args = (k,),method='LSODA',rtol=1e-6,atol=1e-11)
        self.t = np.transpose(sol['t'])
        self.y = np.array(sol['y'])
        if plot_option:
            plt.plot(self.t,np.transpose(self.y))
            plt.xlabel('t')
            plt.ylabel('y')
        return self.t,self.y
    
    def integrate_model_to_ss(self,k=None,y0=None,y0_original=None,tol=1e-17):
        t_max = 10000
        dmet=[1 for i in range(len(self.y0))]
        if y0 is None:
            y0 = copy.deepcopy(self.y0)
        if y0_original is None:
            y0_original = copy.deepcopy(y0)
        while np.dot(dmet,dmet)>tol:
            y0_old = y0
            t,y = self.integrate_model(t_max,k=k,y0=y0)
            y0=self.y[:,-1]
            dmet=(y0-y0_old)
        results = {}
        results['y_ss'] = self.y[:,-1]
        #results['signal'] = (self.y[1,-1]+self.y[4,-1]+self.y[8,-1]+self.y[6,-1])/(y0_original[1]+y0_original[4]+y0_original[8]+y0_original[6])*100
        results['total'] = self.y[1,-1]+self.y[4,-1]+self.y[8,-1]+self.y[6,-1]
        
        results['per_RAS_GTP_Tot'] = (self.y[1,-1]+self.y[4,-1]+self.y[6,-1]+self.y[8,-1])/(y0_original[0]+y0_original[5])*100 # percent RAS-GTP out of total RAS pool
        results['per_RAS_GTP_Eff'] = (self.y[4,-1]+self.y[8,-1])/(y0_original[3])*100 # percent RAS-GTP-Eff out of total Eff pool

        if y0_original[0] > 0:
            results['per_WT_RAS_GTP'] = (self.y[1,-1]+self.y[4,-1])/(y0_original[0])
        else:
            results['per_WT_RAS_GTP'] = 0

        results['per_WT_RAS_GTP_Eff'] = (self.y[4,-1])/(y0_original[3])

        if y0_original[5] > 0:
            results['per_Mut_RAS_GTP'] = (self.y[6,-1]+self.y[8,-1])/(y0_original[5])*100
        else:
            results['per_Mut_RAS_GTP'] = 0
        results['per_Mut_RAS_GTP_Eff'] = (self.y[8,-1])/(y0_original[3])

        results['per_WT_RAS_GTP_Tot'] = (self.y[1,-1]+self.y[4,-1])/(y0_original[0]+y0_original[5])
        results['per_pRAS_GTP_Tot'] = (self.y[6,-1]+self.y[8,-1])/(y0_original[0]+y0_original[5])*100
        results['per_pRAS_Tot'] = (self.y[5,-1]+self.y[6,-1]+self.y[7,-1]+self.y[8,-1])/(y0_original[0]+y0_original[5])*100
        return results
    
    """ OUTDATE - dont use unless update results section
    def solve_ss(self,k=None,y0=None,method='dogbox'):
        if k is None:
            k = copy.deepcopy(self.k)
        if y0 is None:
            y0 = copy.deepcopy(self.y0)
        check_0 = y0[0]+y0[1]+y0[2]+y0[4]
        y0_original = copy.deepcopy(y0)
        #sol = fsolve(self.func,y0,args=(k))
        sol = least_squares(self.func,y0,method=method,args=(k,))['x'] #,bounds=([0 for i in range(len(y0))],[sum for i in range(len(y0))]))
        results = {}
        results['y_ss'] = sol
        #results['signal'] = (sol[1]+sol[4]+sol[8]+sol[6])/(y0_original[1]+y0_original[4]+y0_original[8]+y0_original[6])*100
        results['total'] = sol[1]+sol[4]+sol[8]+sol[6]

        results['per_RAS_GTP_Tot'] = (sol[1]+sol[4]+sol[6]+sol[8])/(y0_original[0]+y0_original[5]) # percent RAS-GTP out of total RAS pool
        results['per_RAS_GTP_Eff'] = (sol[4]+sol[8])/(y0_original[3]) # percent RAS-GTP-Eff out of total Eff pool

        results['per_WT_RAS_GTP_Tot'] = (sol[1]+sol[4])/(y0_original[0]+y0_original[5])
        results['per_WT_RAS_GTP_Eff'] = (sol[4])/(y0_original[3])

        results['per_Mut_RAS_GTP_Tot'] = (sol[6]+sol[8])/(y0_original[0]+y0_original[5])
        results['per_Mut_RAS_GTP_Eff'] = (sol[8])/(y0_original[3])

        check_1 = results['y_ss'][0]+results['y_ss'][1]+results['y_ss'][2]+results['y_ss'][4] # used for checking mass balances when solving for s.s. by least-squares.
        return results
    """
           
    def response_line(self,param,n=50,out_option='total',plot_option=True,solver='integration'):
        param_multipliers = np.logspace(param['range'][0],param['range'][1],n)
        responses = np.zeros(n)
        tq = tqdm(range(n), desc="Running simulations...")
        for m1 in range(n):
                k_modified,y0_modified=self.get_modified_params(params=[param],modifications=[param_multipliers[m1]])
                if solver == 'integration':
                    results = self.integrate_model_to_ss(k=k_modified,y0=y0_modified)
                #elif solver == 'least_squares':
                    #results = self.solve_ss(k=k_modified,y0=y0_modified)
                tq.update()
                try:
                   responses[m1] = results[out_option] #type: ignore
                except:
                    print('error - wrong out_option key for results dict.')
                
        if plot_option:

            plt.plot(param_multipliers,responses)
            plt.xlabel("{}".format(param['name']))
            if out_option == 'signal':
                plt.ylabel('RAS-GTP signal [%]')
            elif out_option == 'total':
                plt.ylabel('total RAS-GTP [M]')
            plt.semilogx()

        else:
            return param_multipliers,responses
    
    def response_surface_2D(self,param_1,param_2,n=50,out_option='total',plot_option=True,solver='integration'):
        param_1_multipliers = np.logspace(param_1['range'][0],param_1['range'][1],n)
        param_2_multipliers = np.logspace(param_2['range'][0],param_2['range'][1],n)
        if param_1['type'] == 'iv':
            param_1_og = self.param_opts[param_1['name']]
        else:
            param_1_og = 1
        if param_2['type'] == 'iv':
            param_2_og = self.param_opts[param_2['name']]
        else:
            param_2_og = 1
        responses = np.zeros([n,n])
        tq = tqdm(range(n*n), desc="Running simulations...")
        for m1 in range(n):
            for m2 in range(n):
                k_modified,y0_modified=self.get_modified_params(params=[param_1,param_2],modifications=[param_1_multipliers[m1],param_2_multipliers[m2]])
                if solver == 'integration':
                    results = self.integrate_model_to_ss(k=k_modified,y0=y0_modified)
                #elif solver == 'least_squares':
                    #results = self.solve_ss(k=k_modified,y0=y0_modified)
                tq.update()
                try:
                   responses[m1,m2] = results[out_option] #type: ignore
                except:
                    print("error - invalid out_option key.")
        if plot_option:
            plt.contourf(np.multiply(param_2_multipliers,param_2_og),np.multiply(param_1_multipliers,param_1_og),responses,levels=25)
            s1 = ''
            s2 = ''
            if param_1['type'] != 'iv':
                s1 = ' multiplier'
            if param_2['type'] != 'iv':
                s2 = ' multiplier'
            plt.xlabel("{}".format(param_2['name']+s2))
            plt.ylabel("{}".format(param_1['name']+s1))
            if out_option == 'signal':
                plt.colorbar(label='RAS-GTP signal [%]')
            elif out_option == 'total':
                plt.colorbar(label='total RAS-GTP [M]')
            plt.semilogx()
            plt.semilogy()

        else:
            return param_2_multipliers,param_1_multipliers,responses
        
    def sobol_analysis_parralell(self,params_to_modify,num_processors=cpu_count(),plot_bar=False,out_option='total',const_param=None,const_mult=None,solver='integration',func=None):
        
        d = len(params_to_modify)
        problem = {
            'num_vars': d,
            'names': [param['name'] for param in params_to_modify],
            'bounds': [param['range'] for param in params_to_modify]
        }
        param_multipliers = saltelli.sample(problem, 1024)
        #print("performing {} {}muM drug simulations".format(1024*2*(d+1),round(drug_dose,4)))
        if const_param is not None:
            params_to_modify = list(params_to_modify)
            params_to_modify.append(const_param)
        inputs = []

        for multiplier in param_multipliers:
            if const_mult is not None:
                multiplier = list(multiplier)
                multiplier.append(const_mult)
            k_new,y0_new = self.get_modified_params(params_to_modify,np.power(10,multiplier))
            inputs.append({'k':k_new,'y0':y0_new,'model':copy.deepcopy(self),'out_option':out_option,'solver':solver,'func':func})

        pool=Pool(processes = num_processors)
        outputs = pool.map(sim_wrapper,inputs)
        pool.close()
        pool.join()

        Y = np.array(outputs)
        Si = sobol.analyze(problem, Y, print_to_console=False)

        if plot_bar:
            plt.bar(problem['names'],Si['ST'])
            plt.xticks(rotation=90)
            plt.title('ST')
            plt.show()
            plt.bar(problem['names'],Si['S1'])
            plt.xticks(rotation=90)
            plt.title('S1')
            plt.show()
            heatmap(Si['S2'], problem['names'], problem['names'])
            plt.title('S2')
            plt.show()

        return problem['names'],Si
    
    def sobol_analysis_parralell_over_param(self,params_to_modify,param_to_vary,n=25,num_processors=cpu_count(),sens_type='ST',out_option='total',solver='integration',title=None,func=None):
        Sis = []
        t = tqdm(range(n), desc="Running simulations...")
        param_range = np.linspace(param_to_vary['range'][0],param_to_vary['range'][1],n)
        for multiplier in param_range:
            names,Si = self.sobol_analysis_parralell(params_to_modify,out_option=out_option,num_processors=num_processors,const_param=param_to_vary,const_mult=multiplier,solver=solver,func=func)
            Sis.append(Si)
            t.update()

        labels = [param['name'] for param in params_to_modify]
        for j in range(len(params_to_modify)):
            signals = []
            for i in range(n):
                signals.append(Sis[i][sens_type][j])
            plt.plot(np.multiply(np.power(10,param_range),self.param_opts[param_to_vary['name']]),signals,label=labels[j])
        plt.xlabel('{}'.format(param_to_vary['name']))
        plt.ylabel('{}'.format(sens_type))
        if title:
            plt.title(title)
        plt.semilogx()
        labelLines()

    def random_parameters_parralell(self,params_to_modify,n=1000,num_processors=cpu_count(),out_option='total',solver='integration',param_multipliers=None):
        inputs = []
        if param_multipliers is None:
            param_multipliers = get_param_multipliers(params_to_modify,n=n)

        for multiplier in param_multipliers:
            k_new,y0_new = self.get_modified_params(params_to_modify,multiplier)
            inputs.append({'k':k_new,'y0':y0_new,'model':copy.deepcopy(self),'out_option':out_option,'solver':solver,'func':None})

        pool=Pool(processes = num_processors)
        outputs = pool.map(sim_wrapper,inputs)
        pool.close()
        pool.join()

        Y = np.array(outputs)
        return Y

def sim_wrapper(input):
    if input['solver'] == 'integration':
        if input['func']:
            results = input['model'].input['func'](k=input['k'],y0=input['y0'])
        else:
            results = input['model'].integrate_model_to_ss(k=input['k'],y0=input['y0'])
    elif input['solver'] == 'least_squares':
        results = input['model'].solve_ss(k=input['k'],y0=input['y0'])
    #results = input['model'].integrate_model_to_ss(k=input['k'],y0=input['y0'])
    if input['out_option'] == 'all_results':
        return results #type: ignore
    try:
        return results[input['out_option']] #type: ignore
    except:
        print('error - invalid out_option key.')

def get_param_multipliers(params_to_modify,n=1000):
    param_multipliers = []
    for i in range(n):
        ss = []
        for param in params_to_modify:
            if param['range_type'] == 'log-uniform':
                s = np.power(10,np.random.default_rng().uniform(param['range'][0],param['range'][1]))
            elif param['range_type'] == 'log-normal':
                s = np.random.lognormal(param['range'][0],param['range'][1],1)
                #s = np.power(10,np.random.default_rng().normal(param['range'][0],param['range'][1]))
            elif param['range_type'] == 'uniform':
                s = np.random.default_rng().uniform(param['range'][0],param['range'][1])
            elif param['range_type'] == 'normal':
                s = np.random.default_rng().normal(param['range'][0],param['range'][1])
            else:
                s = None
            ss.append(s)
        param_multipliers.append(ss)
    return param_multipliers

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="",title=None,show_values=False,colorbar=True,**kwargs):
    #src: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    plt.figure(figsize=(10,10))

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw) #type: ignore
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1), minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_values:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, data[i, j],
                            ha="center", va="center", color="w")
    if title is not None: plt.title(title)

    return im

def bumpchart(df, show_rank_axis= True, rank_axis_distance= 1.1, 
              ax= None, scatter= False, holes= False,
              line_args= {}, scatter_args= {}, hole_args= {},colors=[]):
    
    if ax is None:
        left_yaxis= plt.gca()
    else:
        left_yaxis = ax

    # Creating the right axis.
    right_yaxis = left_yaxis.twinx()
    
    axes = [left_yaxis, right_yaxis]
    
    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)
    
    for i,col in enumerate(df.columns):
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes 
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha= 0)

        left_yaxis.plot(x, y, **line_args, solid_capstyle='round',c=colors[i])
        
        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args,c=colors[i])
            
            #Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color= bg_color, **hole_args)

    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(1, lines + 1)]
    
    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.5, 0.5))
    
    # Sorting the labels to match the ranks.
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index
    
    right_yaxis.set_yticklabels(right_labels)
    
    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance)) #type: ignore
    
    return axes

def lighten_color(color, amount=0.5):
    """
    src: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

