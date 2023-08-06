import ctypes, os, sys

import random

import numpy as np;np.set_printoptions(suppress=True)
from scipy import stats
import matplotlib.pyplot as plt

#from tensorflow import keras 
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, GRU, BatchNormalization
#import tensorflow_addons as tfa
#import tfanightly as tfa

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import model_from_json
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KDTree 
from sklearn.neighbors import BallTree 
from sklearn import neighbors
#from sklearn.neighbors import NearestNeighbors
image_dpi = 175
from tqdm import tqdm
import glob
from datetime import datetime

import Virac as Virac

try:
	import multiprocessing as processing
except:
	import processing



def hyper_params():
	epochs = 128 
	batch_size = 1024
	validation_split = 0.2
	synth = 1
	N = 200
	samps = 1000
	load_model_flag = 0
	make_data = 1
	retrain = 1
	model_parent_dir = '/beegfs/car/njm/models/'
	model_path = model_parent_dir + '/New_data/'
	big_plot_IO = -100
	return epochs,batch_size,validation_split,synth,N,samps,load_model_flag,make_data,retrain,model_parent_dir,model_path,big_plot_IO


def create_model(input_length):
	RNN_Nodes = 1024
	RNN_layers = 12
	model = Sequential()
	#model.add(GRU(RNN_Nodes, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, input_shape = input_length))
	model.add(GRU(RNN_Nodes, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))#, input_shape = input_length))

	for i in range(RNN_layers):
		model.add(GRU(RNN_Nodes, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True))
	
	model.add(GRU(RNN_Nodes, activation='tanh', recurrent_activation='hard_sigmoid'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	#'''
	print ('Compiling...')
	opt = Adam(learning_rate=0.000001)
	#opt = Adam()
	model.compile(loss='binary_crossentropy',
		  optimizer=opt,
		  metrics=['accuracy'])
		  
	return model


def phaser(time, period):
	# this is to mitigate against div 0
	if period == 0:
		period = 1 
	phase = np.array(time) * 0.0
	for i in range(0, len(time)):
		 phase[i] = (time[i])/period - np.floor((time[i])/period)
		 if (phase[i] >= 1):
		   phase[i] = phase[i]-1.
		 if (phase[i] <= 0):
		   phase[i] = phase[i]+1.
	return phase

def gen_chan(mag, phase, knn, N):

	asort  = np.argsort(phase)
	mag = mag[asort]
	phase = phase[asort]
	#print(mag, phase)		
	# Remove NaN values from 'mag' and 'phase'
	mask = ~np.isnan(mag) & ~np.isnan(phase)
	mag = mag[mask]
	phase = phase[mask]
	#print(mag, phase)
	knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
	rn = running_scatter(phase,mag,N)
	delta_phase = []
	prev = 0
	for p in phase:
		delta_phase.append(p - prev)
		prev = p

	return np.vstack((mag, knn_m, rn, phase, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), delta_phase))
	#return np.vstack((mag, smooth(knn_m,int(N/20)), rn, phase, delta_phase))
	#return np.vstack((mag, smooth(knn_m,int(N/5)), rn, phase, delta_phase))
	#return np.vstack((mag, knn_m, rn, phase, delta_phase))
	#return np.vstack((mag, rn, delta_phase, phase))


	
def data_append(mag, phase, knn, N, x_list, y_list, mod):
	x_list.append(gen_chan(mag, phase, knn, N))
	y_list.append(mod)


def get_model(model_path = '/beegfs/car/njm/models/final_12l_dp_all/'):
    #model_path = '/beegfs/car/njm/models/final_better/'
    print("Opening model from here :", model_path)
    json_file = open(model_path+'_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"_model.h5")
    history=np.load(model_path+'_model_history.npy',allow_pickle='TRUE').item()
    model = loaded_model
    N = 200
    knn_N = int(N / 20)
    knn = neighbors.KNeighborsRegressor(knn_N, weights='distance')
    return knn, model



def inference(period, mag, time, knn = None, model = None, N = 200):
    if knn == None or model == None:
        knn, model = get_model()

    if len(mag) < N:
        clip_pad = 1
        pad_length = N - len(mag)
        mag = np.concatenate((mag, mag[:pad_length]))
        time = np.concatenate((time, time[:pad_length]))
    else:
        clip_pad = 0
        mag, mag_shite, time = delete_rand_items(mag, mag, time, len(mag) - N)

    phase = phaser(time, period)
    mag = norm_data(mag)
    FAP = model.predict(np.array([gen_chan(mag, phase, knn, N)]), verbose = 0)
    return FAP[0][0]


def running_scatter(x,y,N):
	rn = []
	xs = np.linspace(min(x),max(x), num = N)
	for i in range(len(xs)):

		if i < 1:
			check = np.where(x < xs[i+2])[0]
		if i == len(xs):
			check = np.where(x >= xs[i-2])[0]
		else:	
			check = np.where((x >= xs[i-2]) & (x <= xs[i]))[0]

		if len(check) > 1: 
			q75, q25 = np.percentile(y[check], [75, 25])
			rn.append(abs(q75-q25))
		else:
			rn.append(0)
	return rn

def smooth(y, box_pts):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

def norm_data(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

def delete_rand_items(mag, magerr, time, n):
	randomlist = random.sample(range(0, len(mag)), n)
	mag = np.array([x for i,x in enumerate(mag) if i not in randomlist])
	magerr = np.array([x for i,x in enumerate(magerr) if i not in randomlist])
	time = np.array([x for i,x in enumerate(time) if i not in randomlist])
	return mag, magerr, time

def delete_rand_items_phase(mag, magerr, time, phase, n):
	randomlist = random.sample(range(0, len(mag)), n)
	mag = np.array([x for i,x in enumerate(mag) if i not in randomlist])
	magerr = np.array([x for i,x in enumerate(magerr) if i not in randomlist])
	time = np.array([x for i,x in enumerate(time) if i not in randomlist])
	phase = np.array([x for i,x in enumerate(phase) if i not in randomlist])
	return mag, magerr, time, phase



###########################################################
#  _	 _	   _	 _		 ____					 #
# | |   (_) __ _| |__ | |_	  / ___|   _ _ ____   _____ #
# | |   | |/ _` | '_ \| __|____| |  | | | | '__\ \ / / _ \#
# | |___| | (_| | | | | ||_____| |__| |_| | |   \ V /  __/#
# |_____|_|\__, |_| |_|\__|	 \____\__,_|_|	\_/ \___|#
#		  |___/										  #
###########################################################


def ax_plotter(axs, mag0, mag1, mag2, mag3, phase, colour):
	N = len(mag1)
	
	axs.plot(np.linspace(0,1, num = N)+1, mag1, c = 'k')
	axs.plot(np.linspace(0,1, num = N), mag1, c = 'k')
	axs.plot(np.linspace(0,1, num = N)+1, mag2, c = 'r')
	axs.plot(np.linspace(0,1, num = N), mag2, c = 'r')
	axs.plot(np.linspace(0,1, num = N)+1, mag3, c = 'gray')
	axs.plot(np.linspace(0,1, num = N), mag3, c = 'gray')
	
	axs.scatter(phase, mag0, marker='x', c = colour)
	axs.scatter(phase+1, mag0, marker='x', c = colour)



def ax_plotter_small(axs, mag, phase, colour):
	
	axs.scatter(phase, mag, marker='x', c = colour)
	axs.scatter(phase+1, mag, marker='x', c = colour)

def lc_debug_plot(lc, data):
	#lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]

	np.set_printoptions(suppress=True)
	plt.clf()

	N = len(lc[1])
	xs = np.linspace(min(lc[0]),max(lc[0]), num = N)
	fig,  ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2)

	ax_plotter_small(ax, lc[0], lc[5], 'grey') 
	ax.errorbar(lc[5], lc[0], yerr = lc[6], ls = 'none')
	ax.errorbar(lc[5]+1, lc[0], yerr = lc[6], ls = 'none')
	ax.set_title("Class:"+str(data[1]))
	ax.set_ylim(0,1)

	ax_plotter_small(ax2, lc[1], xs, 'r') 
	ax_plotter_small(ax2, lc[2], xs, 'g') 
	ax_plotter_small(ax2, lc[3], xs, 'b') 
	ax2.set_ylim(0,1)
	ax2.set_title('Period'+str(round(data[2],4)))

	#ax_plotter_small(ax3, lc[1], lc[5], 'g') 
	ax_plotter(ax3, lc[0], lc[1], lc[2], lc[3], lc[5], 'b') 
	ax3.set_xlabel('A:'+str(round(data[3],3))+'   Err:'+str(round(data[4],3)))
	ax3.set_ylim(0,1)

	ax4.set_xlabel('Sum:'+str(round(sum(lc[4]),3))+'   Med:'+str(round(np.median(lc[4]),3)))
	ax4.scatter(xs, lc[4])
	ax4.set_ylim(0,1)

	plt.savefig(data[0], format = 'png', dpi = image_dpi) 
	plt.clf()
	plt.close()


def LC_train(TOOL, method= 'PDM', N=200):

	def LC_grab(TOOL, method, fp, star_id, period, amplitude, N = 200, mod = 0, synth = 0, EB_mod = 0):

		LC_source_dir = '/beegfs/car/njm/LC/'+fp+'/'+str(int(star_id))+'.FITS'
		light_curve = Virac.fits_open(LC_source_dir)
		#mag, magerr, time = TOOL.error_clip_xy(light_curve["Ks_mag"], light_curve["Ks_emag"], light_curve["Ks_mjdobs"],sigma = 5, err_max = 0.5) #can be clipped
		mag, magerr, time = TOOL.error_clip_xy(light_curve["Ks_mag"],light_curve["Ks_emag"],light_curve["Ks_mjdobs"],light_curve["Ks_ast_res_chisq"],light_curve["Ks_chi"],light_curve["Ks_ambiguous_match"], sigma = 4, err_max = 0.5)
		if len(mag) > 10:
			cat_type = None
			if synth == 1:
				if random.uniform(0,100) > 10:
					amplitude = random.uniform(np.median(magerr)*2,np.median(magerr)*5)
				else:
					amplitude = random.uniform(np.median(magerr)*2,np.median(magerr)*10)
				if EB_mod == 0:
					mag, magerr, time, cat_type, period = TOOL.synthesize(mag = mag, magerr = magerr, time = time, amplitude = amplitude, other_pert = 1, scatter_flag = 1, contamination_flag = 1)
				else:
					mag, magerr, time, cat_type, period = TOOL.synthesize(mag = mag, magerr = magerr, time = time, amplitude = amplitude, other_pert = 1, scatter_flag = 1, contamination_flag = 1, cat_type = 'EB')
					#mag, magerr, time, cat_type, period = TOOL.synthesize(mag = mag, magerr = magerr, time = time, amplitude = amplitude, ceph_LC = 0, eb_LC = 1, rr_LC = 0, yso_LC = 0, other_pert = 1, scatter_flag = 1, contamination_flag = 1)
			if len(mag) < 10:
				return None

			time = np.squeeze(time)
			mag = np.squeeze(mag)
			magerr = np.squeeze(magerr)

			if cat_type == None:
				cat_type = 'Real'
			if mod == 1:
				if random.uniform(0,100) < 10: 
					med_mag = np.median(mag)
					new_mag = []
					if random.uniform(0,100) > 90: 
						med_magerr = np.median(magerr)
						q75, q25 = np.percentile(mag, [75, 25])				
						scatter = abs(q75-q25)/2
						cat_type = "Binary_Error_" + cat_type
						prange = random.choice(TOOL.exclusion_periods)
						period = random.uniform(prange[0],prange[1])
						phase = TOOL.phaser(time, period)	
						asort  = np.argsort(phase)
						mag = mag[asort]
						magerr = magerr[asort]
						time = time[asort]

						mag = ((mag - np.min(mag))/(2*np.max(mag)))+np.min(mag) #normalise so amplitude is 0.5 nomatter what, stops binary from going wild
												
						minplus = random.choice([-1,1])
						for i, m in enumerate(mag):
							mm = abs(m - med_mag)
							if phase[i] > 0.5:
								mm = mm + (scatter * minplus) + random.uniform(med_magerr,3*med_magerr) *0.5
							else:
								mm = mm - (scatter * minplus) + random.uniform(med_magerr,3*med_magerr) *0.5
							new_mag.append(abs(mm+med_mag))
						mag = new_mag
						period = period + (period * random.uniform(-0.1,0.1))
					else:
						cat_type = "High_Scatter_Error_" + cat_type
						for i, m in enumerate(mag):
							new_mag.append(m + random.choice([random.choice([random.uniform(-2,-0.1),random.uniform(0.1,2)]),random.choice([random.gauss(-1,-0.5),random.gauss(1,0.5)])]))
						time = time + (np.random.uniform(0.005,0.015, len(mag)))
						mag = new_mag + np.random.normal(0, magerr*2, len(new_mag))
					if random.uniform(0,100) > 60: 
							prange = random.choice(TOOL.exclusion_periods)
							period = random.uniform(prange[0],prange[1])


				else:
					if random.uniform(0,100) < 50: 
						prange = random.choice(TOOL.exclusion_periods)
						flag = False
						while flag == False:
							if period - (0.1 * period)> prange[0] and period +  (0.1 * period) < prange[1]:
								prange = random.choice(TOOL.exclusion_periods)
							else:
								flag = True
						period = random.uniform(prange[0],prange[1])
					else:
						period = (period * random.uniform(0.3,3) + random.uniform(0.001*np.pi*period,np.pi*period)) * random.uniform(0.977777777777777777, 1.3333333333333333333333)
						random.shuffle(mag)
						time = time + (np.random.uniform(0.005,0.015, len(mag)))*0.1

			time = np.squeeze(time)
			mag = np.squeeze(mag)
			magerr = np.squeeze(magerr)
		
			timee = time
			if len(mag) < N:
				clip_pad = 1
				mag = np.pad(mag, (0,int((N - len(mag)))), 'wrap')
				magerr = np.pad(magerr, (0,int((N - len(magerr)))), 'wrap')
				time = np.pad(time, (0,int((N - len(time)))), 'wrap')
			else:
				clip_pad = 0
				mag, magerr, time = delete_rand_items(mag, magerr, time, len(mag)-N)
				
			time = np.squeeze(time)
			mag = np.squeeze(mag)
			magerr = np.squeeze(magerr)

			try:
				phase = TOOL.phaser(time, period)
			except:
				print(time)
				print(timee)
				exit()

			phase2 = TOOL.phaser(time, period + (0.00001 * period * random.choice([-1,1]) * np.random.uniform(0,1)))

			phase3 = TOOL.phaser(time, period/2)

			phase4 = TOOL.phaser(time, period + (0.00001 * period * random.choice([-1,1]) * np.random.uniform(0,1)))

			if 'EB' in cat_type or 'CV' in cat_type:
					phase4 = phase
					phase3 = phase

			if EB_mod == 1:
				phase3 = TOOL.phaser(time, period + (0.00001 * period * random.choice([-1,1]) * np.random.uniform(0,1)))

			#NN NEEDS TO SEE DBOULE PERIOD ATLEAST, IT KEEPS REJECTING THEM
			return mag, magerr, phase, phase2, phase3, phase4, cat_type, period, clip_pad
		else:
			return None


	def phase_shift(phase, mod = 0):
		phase = phase + (random.uniform(0.1,1)*random.choice([-1,1]))
		phase_mask = np.where(phase > 1)[0]
		phase[phase_mask] = phase[phase_mask] - 1
		phase_mask = np.where(phase < 0)[0]
		phase[phase_mask] = phase[phase_mask] + 1
		if mod == 1:
			#random.shuffle(phase)
			pass
		return phase

	def data_add(TOOL, method, fp, name, period, amplitude, knn, N = 200, mod = 0, synth = 0, train = 1, EB_mod = 0):


		flag = False
		try:
			Mag, Magerr, Phase, Phase2, Phase3, Phase4, cat_type, period, clip_pad = LC_grab(TOOL, method, fp, name, period, amplitude, N = N, mod = mod, synth = synth, EB_mod = EB_mod)
			flag = True
		except Exception as e:
			print(e)
			print(period)
			print(name)
			print(amplitude)
			flag = False
	
		if flag:
			for noise_mult in [0]:#,1]:#,2]:
				mag = Mag
				magerr = Magerr
				phase = Phase + (np.random.uniform(0.005,0.015, len(mag)) * noise_mult)*0.0001
				phase2 = Phase2 + (np.random.uniform(0.005,0.015, len(mag)) * noise_mult)*0.0001
				phase3 = Phase3 + (np.random.uniform(0.005,0.015, len(mag)) * noise_mult)*0.0001
				mag = mag + (np.random.normal(0, magerr*0.1, len(mag)) * noise_mult)*0.001
				
				db_flag = 0
				if int(np.random.uniform(0,100)) < 0:# and mod == 2:
					db_flag = 1			
				mag = norm_data(mag)

				if train == 1:

					data_append(mag, phase, knn, N, x_train, y_train, mod)
					if db_flag == 1:
						knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
						rn = running_scatter(phase,mag,N)
						lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
						data = ['/beegfs/car/njm/OUTPUT/vars/debug/'+cat_type+'_nm'+str(noise_mult) + '_p' + '1' +  '_m' +  str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
						lc_debug_plot(lc, data)

					phase_shifts = 10
					for x in range(phase_shifts):
						phase = phase_shift(phase, mod)
						data_append(mag, phase, knn, N, x_train, y_train, mod)
						if db_flag == 1:
							knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
							rn = running_scatter(phase,mag,N)
							lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
							data = ['/beegfs/car/njm/OUTPUT/vars/debug/'+cat_type+'_nm'+str(noise_mult) + '_p'+str(x)+ '_m' + str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
							lc_debug_plot(lc, data)

					phase = phase2
					phase_shifts = 5
					for x in range(phase_shifts):
						phase = phase_shift(phase, mod)
						data_append(mag, phase, knn, N, x_train, y_train, mod)
						if db_flag == 1:
							knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
							rn = running_scatter(phase,mag,N)
							lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
							data = ['/beegfs/car/njm/OUTPUT/vars/debug/'+cat_type+'_nm'+str(noise_mult) + '_p2'+str(x)+ '_m' + str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
							lc_debug_plot(lc, data)

					phase = Phase4
					for x in range(phase_shifts):
						phase = phase_shift(phase, mod)
						data_append(mag, phase, knn, N, x_train, y_train, mod)
						if db_flag == 1:
							knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
							rn = running_scatter(phase,mag,N)
							lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
							data = ['/beegfs/car/njm/OUTPUT/vars/debug/'+cat_type+'_nm'+str(noise_mult) + '_p3'+str(x)+ '_m' + str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
							lc_debug_plot(lc, data)
			
				else:

					knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
					data_append(mag, phase, knn, N, x_test, y_test, mod)
					if db_flag == 1:
						knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
						rn = running_scatter(phase,mag,N)
						lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
						data = ['/beegfs/car/njm/OUTPUT/vars/debug/test/'+cat_type+'_nm'+str(noise_mult) + '_p' + '1' +  '_m' +  str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
						lc_debug_plot(lc, data)

					phase_shifts = 10
					for x in range(phase_shifts):
						phase = phase_shift(phase, mod)
						data_append(mag, phase, knn, N, x_test, y_test, mod)
						if db_flag == 1:
							knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
							rn = running_scatter(phase,mag,N)
							lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
							data = ['/beegfs/car/njm/OUTPUT/vars/debug/test/'+cat_type+'_nm'+str(noise_mult) + '_p'+str(x)+ '_m' + str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
							lc_debug_plot(lc, data)

					phase = phase3
					data_append(mag, phase, knn, N, x_test, y_test, mod)
					if db_flag == 1:
						knn_m = knn.fit(phase[:, np.newaxis], mag).predict(np.linspace(0,1, num = N)[:, np.newaxis])
						rn = running_scatter(phase,mag,N)
						lc = [mag, knn_m, smooth(knn_m,int(N/20)), smooth(knn_m,int(N/5)), rn, phase, magerr]
						data = ['/beegfs/car/njm/OUTPUT/vars/debug/test/'+cat_type+'_nm'+str(noise_mult) + '_p3' + '1' +  '_m' +  str(mod) +  '_cp' +  str(clip_pad) + '.png',cat_type, period, amplitude, np.median(magerr)]
						lc_debug_plot(lc, data)






	#==============#
	# HYPER PARAMS #
	#==============#
	epochs,batch_size,validation_split,synth,N,samps,load_model_flag,make_data,retrain,model_parent_dir,model_path,big_plot_IO = hyper_params()

	#TODO normalise training set, and validation set. 

	TOOL.tempio = TOOL.IO
	TOOL.IO = 0
	x_train = []; y_train = []

	knn_N = int(N / 20)
	knn = neighbors.KNeighborsRegressor(knn_N, weights='distance', p = 2)

	if os.path.exists(model_path) == False:
		os.mkdir(model_path)
	if load_model_flag == 1:
		json_file = open(model_path+'_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(model_path+"_model.h5")
		history=np.load(model_path+'_model_history.npy',allow_pickle='TRUE').item()
		model = loaded_model
	else:

		if TOOL.IO > -2:
			print('\t~~~~~~~~~~~LC-LSTM~~~~~~~~~~~')
			print("\tMethod :", method)
			print("\tN :", N)
			print("\tEpochs :", epochs)
			print("\tBatch Size :", batch_size)
			print("\tValidation Split :", validation_split)
			print('\t~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')





	if make_data == 1:

		phase_shifts = 25
		TOOL.OUTPUT_redux_load(load_file_path = '/beegfs/car/njm/OUTPUT/vars/Redux_vars_filled.csv')

		#================#
		# ADD REAL P     #
		#================#
		tier = "/beegfs/car/njm/Periodic_Variables/Best/Figures/Light_Curve/*.jpg"
		files = glob.glob(tier)
		random.shuffle(files)
		files = files[:int(len(files)*0.1)]

		print("Adding this many real periodics:",len(files))

		for fi in tqdm(files):
			star_name = fi.split('/')[-1].replace('__','_')
			star_name = star_name.split('.')[0]#[:-4]
			mag, magerr, phase, time = np.genfromtxt("/beegfs/car/njm/Periodic_Variables/Best/LC/"+star_name+".csv", dtype='float', converters = None, comments = '#', delimiter = ',').T
			for x in range(phase_shifts):
				phase = phase_shift(phase)

				if len(mag) < N:
					clip_pad = 1
					mag = np.pad(mag, (0,int((N - len(mag)))), 'wrap')
					magerr = np.pad(magerr, (0,int((N - len(magerr)))), 'wrap')
					time = np.pad(time, (0,int((N - len(time)))), 'wrap')
					phase = np.pad(phase, (0,int((N - len(phase)))), 'wrap')

				else:
					clip_pad = 0
					mag, magerr, time, phase = delete_rand_items_phase(mag, magerr, time, phase, len(mag)-N)
					
				data_append(mag, phase, knn, N, x_train, y_train, 0)

		#================#
		#ADD FAKE EB DATA#
		#================#
		synth_samples = 2000
		tier = "/beegfs/car/njm/Periodic_Variables/Figures/Light_Curve/*.jpg"
		files = glob.glob(tier)
		random.shuffle(files)
		files = files[:synth_samples]
		print("Adding this many fake ebs:",len(files))
		for fi in tqdm(files):
			star_name = fi.split('/')[-1].replace('__','_')
			star_name = star_name.split('.')[0]#[:-4]
			idx = np.where(int(star_name) == TOOL.list_name)[0]
			TOOL.OUTPUT_redux_index_assign(idx)
			data_add(TOOL, method, 'vars', star_name, TOOL.true_period, TOOL.true_amplitude, knn, N = N, mod = 0, synth = 1, EB_mod = 1)

		#===================#
		#ADD FAKE SYNTH DATA#
		#===================#
		tier = "/beegfs/car/njm/Periodic_Variables/Figures/Light_Curve/*.jpg"
		files = glob.glob(tier)
		random.shuffle(files)
		files = files[:synth_samples]
		print("Adding this many fake periodics:",len(files))
		for fi in tqdm(files):
			star_name = fi.split('/')[-1].replace('__','_')
			star_name = star_name.split('.')[0]#[:-4]
			idx = np.where(int(star_name) == TOOL.list_name)[0]
			TOOL.OUTPUT_redux_index_assign(idx)
			data_add(TOOL, method, 'vars', star_name, TOOL.true_period, TOOL.true_amplitude, knn, N = N, mod = 0, synth = 1, EB_mod = 0)

		periodics = len(y_train)





 
		#================#
		#   ADD REAL AP  #
		#================#
		tier = "/beegfs/car/njm/Aperiodic_Variables/Figures/Light_Curve/*.csv"
		files = glob.glob(tier)
		random.shuffle(files)
		files = files[:int(len(files)*0.01)]
		print("Adding this many real aperiodics:",len(files))
		for fi in tqdm(files):
			star_name = fi.split('/')[-1].replace('__','_')
			star_name = star_name.split('.')[0]#[:-4]
			mag, magerr, phase, time = np.genfromtxt("/beegfs/car/njm/Aperiodic_Variables/LC/"+star_name+".csv", dtype='float', converters = None, comments = '#', delimiter = ',').T
			for x in range(phase_shifts):
				phase = phase_shift(phase)

				if len(mag) < N:
					clip_pad = 1
					mag = np.pad(mag, (0,int((N - len(mag)))), 'wrap')
					magerr = np.pad(magerr, (0,int((N - len(magerr)))), 'wrap')
					time = np.pad(time, (0,int((N - len(time)))), 'wrap')
					phase = np.pad(phase, (0,int((N - len(phase)))), 'wrap')

				else:
					clip_pad = 0
					mag, magerr, time, phase = delete_rand_items_phase(mag, magerr, time, phase, len(mag)-N)
					
				data_append(mag, phase, knn, N, x_train, y_train, 1)


		aperiodics = len(y_train) - periodics

		#===================#
		#ADD FAKE SYNTH DATA#
		#===================#
		synth_samples = periodics - aperiodics
		tier = "/beegfs/car/njm/Aperiodic_Variables/Figures/Light_Curve/*.jpg"
		files = glob.glob(tier)
		random.shuffle(files)
		files = files[:synth_samples]
		print("Adding this many fake aperiodics:",len(files))
		for fi in tqdm(files):
			star_name = fi.split('/')[-1].replace('__','_')
			star_name = star_name.split('.')[0]#[:-4]
			idx = np.where(int(star_name) == TOOL.list_name)[0]
			TOOL.OUTPUT_redux_index_assign(idx)
			data_add(TOOL, method, 'vars', star_name, TOOL.true_period, TOOL.true_amplitude, knn, N = N, mod = 1, synth = 1, EB_mod = 0)

		y_train = np.array(y_train)
		x_train = np.array(x_train)

		#np.savez(model_parent_dir + '/data/DATA_'+str(real_samples)+'_'+str(ebsig_100_samples)+str(sig_100_samples)+'_'+str(d001_samples)+'_'+str(d002_samples)+'.npz', x=x_train, y=y_train)

		np.savez(model_parent_dir + '/data/DATA_'+str(method)+'.npz', x=x_train, y=y_train)



		if method == 'vars_0':
			np.savez(model_parent_dir + '/data/DATA.npz', x=x_train, y=y_train)
			print('Loading Test Data')
			#================#
			#ADD TESTING DATA#
			#================#
			fp = 'd002'
			SAMPLE_FILE = np.loadtxt('/beegfs/car/njm/useless_OUTPUT/d002_LSTM/d002_LSTM.csv', comments = '#', delimiter = ",").T
			SAMPLE_NAME = SAMPLE_FILE[0]
			SAMPLE_PERIOD = SAMPLE_FILE[7]
			SAMPLE_AMPLITUDE = SAMPLE_FILE[8]		#THIS WILL BREAK WHEN OUTPUT IS CHANGED!!!!!!!!!!!!!!!!!!!!
			test_samples = 4000#int(len(SAMPLE_NAME)*0.01)
			synth = 1
			indexes = list(range(0,test_samples))
			random.shuffle(indexes)
			x_test = []; y_test = []
			for ii in range(test_samples):
				i = indexes[ii]
				TOOL.name = method + str(i)
				for mod in [0,1]:
					period = SAMPLE_PERIOD[i]
					name = SAMPLE_NAME[i]
					amplitude = SAMPLE_AMPLITUDE[i]
					data_add(TOOL, method, fp, name, period, amplitude, knn, N = N, mod = mod, synth = synth, train = 0)

			sig_100_fp = '100_sig_vars'
			sig_100_path = '/beegfs/car/njm/useless_OUTPUT/100_sig_vars/100_sig_vars.csv'
			sig_100_SAMPLE_FILE = np.loadtxt(sig_100_path, comments = '#', delimiter = ",", usecols = [0,8,29], skiprows = 1).T
			sig_100_names = sig_100_SAMPLE_FILE[0]
			sig_100_amplitudes = sig_100_SAMPLE_FILE[1]		#THIS WILL BREAK WHEN OUTPUT IS CHANGED!!!!!!!!!!!!!!!!!!!!
			sig_100_periods = sig_100_SAMPLE_FILE[2]
			test_samples = 4000#int(len(SAMPLE_NAME)*0.01)
			synth = 1
			indexes = list(range(0,test_samples))
			random.shuffle(indexes)
			x_test = []; y_test = []
			for ii in range(test_samples):
				i = indexes[ii]
				TOOL.name = method + str(i)
				for mod in [0,1]:
					period = SAMPLE_PERIOD[i]
					name = SAMPLE_NAME[i]
					amplitude = SAMPLE_AMPLITUDE[i]
					data_add(TOOL, method, fp, name, period, amplitude, knn, N = N, mod = mod, synth = synth, train = 0)

			y_test = np.array(y_test)
			x_test = np.array(x_test)
			np.savez(model_parent_dir + '/data/TEST_DATA.npz', x=x_train, y=y_train)
		exit()

	else:
		TOOL.epoch = 0 #allow saving of last epoch
		def LR_Sched(epoch):
			linear_step = 6 

			if TOOL.epoch > 30:
				TOOL.epoch = TOOL.epoch-5

			if TOOL.epoch < epoch:
				TOOL.epoch = epoch
			else:
				epoch = TOOL.epoch + epoch + 1

			if epoch < linear_step:
				lr =  0.00001
			else:
				if epoch > 15:
					lr =  0.00001 * (0.7 ** ((epoch - linear_step)))
				else:
					lr =  0.00001 * (0.9 ** ((epoch - linear_step)))
			print(lr,'	',epoch,'!!!!!!')
			return lr

		model = create_model(200)
		lrate = LearningRateScheduler(LR_Sched)
		es = EarlyStopping(monitor='val_loss', mode='min', min_delta = 0.00001, verbose=1, patience=5)


		with np.load(model_parent_dir + '/data/TEST_DATA.npz') as data:
			x_test = data['x']
			y_test = data['y']


		for method in ['vars_0','vars_1','vars_2','vars_3','vars_4','vars_5','vars_6','vars_7','vars_8','vars_9','vars_10','vars_11','vars_12','vars_13','vars_14','vars_15']:

			with np.load(model_parent_dir + '/data/DATA_'+str(method)+'.npz') as data:
				x_train = data['x']
				y_train = data['y']


			print('Training on this many samples:', len(y_train)/2)
			TOOL.IO = TOOL.tempio
			if TOOL.IO > big_plot_IO:
				bigioplot(x_train, y_train, model_path+'/'+method+'_INPUT.png')


			if retrain == 1 or load_model_flag == 0:

				history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split = validation_split, verbose = 1, shuffle = True, workers = 100, use_multiprocessing = True, callbacks=[es, lrate])
				history = history.history


			if TOOL.IO > big_plot_IO:
				test_loss, test_acc = model.evaluate(x_test, y_test)

				print('\t Model Accuracy : ', test_acc)
				print('\t Model Loss : ', test_loss)

				metric = "accuracy"
				plt.figure()
				plt.plot(history[metric])
				plt.plot(history["val_" + metric])
				plt.title("Tested Accuracy :" + str(round(test_acc,3)))
				plt.ylabel('Accuracy')
				plt.xlabel("Epoch", fontsize="large")
				plt.legend(["train", "val"], loc="best")
				plt.savefig(model_path+'/'+method+'_TEST_ACC.png', format = 'png', dpi = image_dpi) 
				plt.close()

				plt.plot(history['loss'])
				plt.plot(history['val_loss'])
				plt.title("Tested Loss :" + str(round(test_loss,3)))
				plt.ylabel('Loss')
				plt.xlabel('Epoch')
				plt.legend(['train', 'val'], loc='upper left')
				plt.savefig(model_path+'/'+method+'_TEST_LOSS.png', format = 'png', dpi = image_dpi) 
			
		if retrain == 1 or load_model_flag == 0:
			model_json = model.to_json()
			with open(model_path+"_model.json", "w") as json_file:
				json_file.write(model_json)

			model.save_weights(model_path+"_model.h5")
			np.save(model_path+'_model_history.npy',history)
			print("Saved model to disk")



	if TOOL.IO > big_plot_IO:
		if TOOL.IO > big_plot_IO:
			bigioplot(x_test, y_test, model_path+'/'+method+'_OUTPUT.png')


		print('\t~~~~~~~~~LSTM-TEST~~~~~~~~')
		print("\tMethod :", method)
		print("\tSamples :", samples)
		print("\tTest accuracy:", test_acc)
		print("\tTest loss:", test_loss)
		print("\tP FAP 1:",pfap1)
		print("\tP FAP 2:",pfap2)
		print("\tP FAP 3:",pfap3)
		print("\tP FAP 4:",pfap4)
		print("\tP FAP 5:",pfap5)
		print("\tP FAP 6:",pfap6)
		print('\t--------------------------')
		print("\tAP FAP 1:",apfap1)
		print("\tAP FAP 2:",apfap2)
		print("\tAP FAP 3:",apfap3)
		print("\tAP FAP 4:",apfap4)
		print("\tAP FAP 5:",apfap5)
		print("\tAP FAP 6:",apfap6)
		print('\t~~~~~~~~~~~~~~~~~~~~~~~~~~')

#LC_inference(TOOL, model = model, N = N)




def LC_inference(TOOL, model = None, model_path = None,  N = 200):
	round_to_n = lambda x, n: x if x == 0 else round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))
	if model == None:
		if model_path == None:
			model_path = '/beegfs/car/njm/models/final_12l_dp_all/'
		json_file = open(model_path+'_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(model_path+"_model.h5")
		history=np.load(model_path+'_model_history.npy',allow_pickle='TRUE').item()
		model = loaded_model

	knn_N = int(N / 20)
	knn = neighbors.KNeighborsRegressor(knn_N, weights='distance')
	TOOL.redux = 3
	TOOL.csv_dir = 'Redux_'+ TOOL.csv_dir
	TOOL.OUTPUT_redux_load()
	#TOOL.OUTPUT_load()
	x_test = [];y_test = []
	TOOL.tempio = TOOL.IO
	TOOL.IO = 0
	TOOL.error_clip = 1
	TOOL.s_fit = 1
	new_continue = ''#'new' #This is not yet in the function imports. ONLY DEFINED HERE

	try:
		if new_continue == 'new':
			now = datetime.now()
			new_name = TOOL.output_dir+'FAP_'+str(now.strftime("%H%M%S"))+'_'+TOOL.csv_dir
			old_name = TOOL.output_dir+'FAP_'+TOOL.csv_dir
			os.rename(old_name, new_name)
			finishedid = []
			print('Old file named to:', new_name)
		else:
			print('Opening previous work...')
			finishedid= np.genfromtxt(TOOL.output_dir+'FAP_'+TOOL.csv_dir, comments = '#', delimiter = ",", skip_header = 1, usecols = [0]).T
			print('...Opened!')
	except Exception as e:
		finishedid = []
		print(repr(e))
		#exit()


	for i, sname in enumerate(TOOL.list_name):
		if i % 1000 == 0:
			print(i, len(TOOL.list_name))
		if int(sname) not in map(int,finishedid):
			TOOL.OUTPUT_redux_index_assign(i)
			#TOOL.OUTPUT_index_assign(i)

			FAP_list = []
			lightcurve = Virac.fits_open('/beegfs/car/njm/LC/'+TOOL.data_name+'/'+str(int(TOOL.name))+'.FITS')
			TOOL.lightcurve(lightcurve["Ks_sourceid"][0],lightcurve["Ks_mag"],lightcurve["Ks_emag"],lightcurve["Ks_mjdobs"])

			if len(TOOL.mag) < N:
				new_mag = np.pad(TOOL.mag, (0,int((N - len(TOOL.mag)))), 'wrap')
				new_magerr = np.pad(TOOL.magerr, (0,int((N - len(TOOL.magerr)))), 'wrap')
				new_time = np.pad(TOOL.time, (0,int((N - len(TOOL.time)))), 'wrap')
			else:
				new_mag, new_magerr, new_time = delete_rand_items(TOOL.mag,TOOL.magerr,TOOL.time,len(TOOL.mag)-N)

			new_mag[new_mag < 2] = np.median(new_mag)

			if TOOL.ls_p == 0: 
				TOOL.ls_fap = 1
			else:
				TOOL.ls_fap = inference(TOOL.ls_p, new_mag, new_time, knn, model, TOOL)

			if TOOL.pdm_p == 0: 
				TOOL.pdm_fap = 1
			else:
				TOOL.pdm_fap = inference(TOOL.pdm_p, new_mag, new_time, knn, model, TOOL)

			if TOOL.ce_p == 0: 
				TOOL.ce_fap = 1
			else:
				TOOL.ce_fap = inference(TOOL.ce_p, new_mag, new_time, knn, model, TOOL)

			if TOOL.gp_p == 0: 
				TOOL.gp_fap = 1
			else:
				TOOL.gp_fap = inference(TOOL.gp_p, new_mag, new_time, knn, model, TOOL)

			if TOOL.true_period == 0: 
				TOOL.true_fap = 1
			else:
				TOOL.true_fap = inference(TOOL.true_period, new_mag, new_time, knn, model, TOOL)


			FAPS = np.array([TOOL.ls_fap, TOOL.pdm_fap, TOOL.ce_fap, TOOL.gp_fap, TOOL.true_fap])
			PERIODS = [TOOL.ls_p, TOOL.pdm_p, TOOL.ce_p, TOOL.gp_p, TOOL.true_period]			
			FAP_LIMITS = [0.05, 0.01, 0.001, 0.0001, 0.00001, 0.0000001]


			continue_flag = 0
			if sum(FAPS) - FAPS[-1] > len(FAPS) - FAPS[-1] -0.9: #rough check
				continue_flag = 1
				TOOL.true_class = "Aperiodic"

			else:
				new_period = PERIODS[np.argmin(FAPS)]
				TOOL.true_period = new_period
				TOOL.true_class = TOOL.dtw_class(new_time, new_period, new_mag, knn)

				ep_flag = 0

				for ep in TOOL.exclusion_periods:
					if new_period > ep[0] and new_period < ep[1]:
						ep_flag = 1 


				while continue_flag == 0 and ep_flag == 0:

					confirmed = 0
					if sum(np.where(FAP_LIMITS[0] > FAPS)[0]) > 1:	#Check if more than 1 period gives low FAP
						confirmed = 1

					FAP_LEVEL = 0
					for i, limit in enumerate(FAP_LIMITS):	#Find lowest fap
						if min(FAPS) < limit:
							FAP_LEVEL = i

					if min(FAPS) < 0.0001:
						unique_periods =  []

						if confirmed == 1:
							#fp = TOOL.light_curve_figure+'/'+str(FAP_LEVEL)+'/Confirmed/'+TOOL.name
							fp = TOOL.light_curve_figure+'/Confirmed/FAP_'+str(FAP_LEVEL)+'_'+TOOL.name
						else:
							#fp = TOOL.light_curve_figure+'/'+str(FAP_LEVEL)+'/Unconfirmed/'+TOOL.name
							fp = TOOL.light_curve_figure+'/Unconfirmed/FAP_'+str(FAP_LEVEL)+'_'+TOOL.name

						TOOL.folded_lc_true(fp = fp, true_fap = min(FAPS))
					continue_flag = 1
			TOOL.OUTPUT_write(update_flag = 3)
		TOOL.IO = TOOL.tempio




def bigioplot(x_train, y_train, fp):


	samples = len(y_train)/2
	s10 = samples / 10

	pos_1=abs(int((samples)/10) + int(random.uniform(-s10,+s10)))
	pos_2=abs(int((samples)/9) + int(random.uniform(-s10,+s10)))
	pos_3=abs(int((samples)/10)*2 + int(random.uniform(-s10,+s10)))
	pos_4=abs(int((samples)/10)*3 + int(random.uniform(-s10,+s10)))
	pos_5=abs(int((samples)/10)*4 + int(random.uniform(-s10,+s10)))
	pos_6=abs(int((samples)/10)*5 + int(random.uniform(-s10,+s10)))
	pos_7=abs(int((samples)/10)*6 + int(random.uniform(-s10,+s10)))
	pos_8=abs(int((samples)/10)*7 + int(random.uniform(-s10,+s10)))
	pos_9=abs(int((samples)/10)*8 + int(random.uniform(-s10,+s10)))
	pos_10=abs(int((samples)/10)*9 + int(random.uniform(-s10,+s10)))

	classes = np.unique(y_train)
	a_per1 = x_train[np.where(np.array(y_train) == 1)[0][pos_1]]
	per1 = x_train[np.where(np.array(y_train) == 0)[0][pos_1]]
	a_per2 = x_train[np.where(np.array(y_train) == 1)[0][pos_2]]
	per2 = x_train[np.where(np.array(y_train) == 0)[0][pos_2]]
	a_per3 = x_train[np.where(np.array(y_train) == 1)[0][pos_3]]
	per3 = x_train[np.where(np.array(y_train) == 0)[0][pos_3]]
	a_per4 = x_train[np.where(np.array(y_train) == 1)[0][pos_4]]
	per4 = x_train[np.where(np.array(y_train) == 0)[0][pos_4]]
	a_per5 = x_train[np.where(np.array(y_train) == 1)[0][pos_5]]
	per5 = x_train[np.where(np.array(y_train) == 0)[0][pos_5]]
	a_per6 = x_train[np.where(np.array(y_train) == 1)[0][pos_6]]
	per6 = x_train[np.where(np.array(y_train) == 0)[0][pos_6]]
	a_per7 = x_train[np.where(np.array(y_train) == 1)[0][pos_7]]
	per7 = x_train[np.where(np.array(y_train) == 0)[0][pos_7]]
	a_per8 = x_train[np.where(np.array(y_train) == 1)[0][pos_8]]
	per8 = x_train[np.where(np.array(y_train) == 0)[0][pos_8]]
	a_per9 = x_train[np.where(np.array(y_train) == 1)[0][pos_9]]
	per9 = x_train[np.where(np.array(y_train) == 0)[0][pos_9]]
	a_per10 = x_train[np.where(np.array(y_train) == 1)[0][pos_10]]
	per10 = x_train[np.where(np.array(y_train) == 0)[0][pos_10]]

	# == == == == == == == == == == == == == == == == = 
	#plot the folded light curve
	# == == == == == == == == == == == == == == == == = 

	plt.clf()

	fig, ((ap1_ax, ap2_ax, ap3_ax, p1_ax, p2_ax, p3_ax),(ap4_ax, ap5_ax, ap6_ax, p4_ax, p5_ax, p6_ax),(ap7_ax, ap8_ax, ap9_ax, p7_ax, p8_ax, p9_ax)) = plt.subplots(3, 6, figsize=(16, 8))

	np.set_printoptions(suppress=True)


	# == == == == == == == == == == = 
	#A Periodic 1
	# == == == == == == == == == == =
	data = a_per1
	ax_plotter_small(ap1_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap1_ax.set_title('AP')
	ap1_ax.xaxis.tick_top()
	ap1_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap1_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap1_ax.set_xticklabels([])

	data = a_per2
	ax_plotter_small(ap2_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap2_ax.set_title('AP')
	ap2_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap2_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap2_ax.set_xticklabels([])

	data = a_per3
	ax_plotter_small(ap3_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap3_ax.xaxis.tick_top()
	ap3_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap3_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap3_ax.set_xticklabels([])

	data = a_per4
	ax_plotter_small(ap4_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap4_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap4_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap4_ax.set_xticklabels([])

	data = a_per5
	ax_plotter_small(ap5_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap5_ax.xaxis.tick_top()
	ap5_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap5_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap5_ax.set_xticklabels([])

	data = a_per6
	ax_plotter_small(ap6_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap6_ax.xaxis.tick_top()
	ap6_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap6_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap6_ax.set_xticklabels([])

	data = a_per7
	ax_plotter_small(ap7_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap7_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap7_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap7_ax.set_xticklabels([])
			
	data = a_per8
	ax_plotter_small(ap8_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap8_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap8_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap8_ax.set_xticklabels([])

	data = a_per9
	ax_plotter_small(ap9_ax, data[0], data[3], 'b')#data[2], data[3],data[5], 'b')  
	ap9_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	ap9_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	ap9_ax.set_xticklabels([])

	# == == == == == == == == == == = 
	#Periodic 1
	# == == == == == == == == == == = 
	data = per1
	ax_plotter_small(p1_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p1_ax.set_title('P')
	p1_ax.xaxis.tick_top()
	p1_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p1_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p1_ax.set_xticklabels([])

	data = per2
	ax_plotter_small(p2_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p2_ax.set_title('P')
	p2_ax.xaxis.tick_top()
	p2_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p2_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p2_ax.set_xticklabels([])

	data = per3
	ax_plotter_small(p3_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p3_ax.xaxis.tick_top()
	p3_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p3_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p3_ax.set_xticklabels([])

	data = per4
	ax_plotter_small(p4_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p4_ax.xaxis.tick_top()
	p4_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p4_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p4_ax.set_xticklabels([])

	data = per5
	ax_plotter_small(p5_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p5_ax.xaxis.tick_top()
	p5_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p5_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p5_ax.set_xticklabels([])

	data = per6
	ax_plotter_small(p6_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p6_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p6_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p6_ax.set_xticklabels([])

	data = per7
	ax_plotter_small(p7_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p7_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p7_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p7_ax.set_xticklabels([])
			
	data = per8
	ax_plotter_small(p8_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p8_ax.xaxis.tick_top()
	p8_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p8_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p8_ax.set_xticklabels([])

	data = per9
	ax_plotter_small(p9_ax, data[0], data[3], 'g')# data[2], data[3],data[5], 'g') 
	p9_ax.xaxis.tick_top()
	p9_ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
	p9_ax.tick_params(axis = "y", which = "both", right = False, left = False)
	p9_ax.set_xticklabels([])
	plt.savefig(fp, format = 'png', dpi = image_dpi) 
	plt.clf()

