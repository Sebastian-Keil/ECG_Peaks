import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import heartpy as hp
import os
from tqdm import tqdm 
import mne
import datetime 

# Enter the participant numbers you wish to process here format use just the three 
# digit number in quotationmarks eg.: 017 or 017_EOT separated by commas
codes = ["<code1>", "<code2>"] # This obviously allows for more participants and tasks than two each
tasks = ["<task1>", "<task2>"]
now = datetime.datetime.now()
output = "<path to output>"
sample_rate = 1000  # This is the target sampling rate NOT the sampling you used to record the signal!

with open(output + 'log.txt', 'a') as f:
    print(f'\n New run at {now}', file=f)
# The code loops over each of your participants and conditions
for code in tqdm(codes, desc="Participants"):
    for task in tqdm(tasks, desc="Tasks", leave=False):
        if task == "<task1>": # You can remove this if else block if you want to just use the task variables to indentify conditions
            out = "<out>"
        else:
            out = "<out2>"
    # This path needs to be adjusted to whatever the final folder structure will be 
    # this is only useful if you have different folders for T1 and T2 measurements
        if len(code) == 3 :
            path = f"<path1>{code}"
        else:
            short_code = code[:-4]
            path = f"<path2>{short_code}"
    # Removing previously generated files if there are any
        try:    
            os.remove(output + f"/sub-{code}/beh/sub_{code}-{out}.csv")
        except FileNotFoundError:
            print('No file to delete')
    # This routine checks for the availability of a start marker
    # if one is found the variable is set to crop out anything before and 
    # then process the following 15 Minutes of SIA
        try:
            with open(path + f"/sub_{code}_{task}.vmrk", 'r') as file:
                start=0
                stop=900
                lines = file.read()
                start = int(lines.split("Mk2=Stimulus,S  1,")[1].split(',')[0])
                #This needs to be adjusted for whatever sampling rate you were using
                start = round(start /5000)                                     
                stop += start
        except IndexError:
            with open(output + 'log.txt', 'a') as f:
                print(f'sub_{code}_{task}: No Marker found, proceeding from start of file', file=f)
            start=0
            stop=900
        except FileNotFoundError:
            with open(output + 'log.txt', 'a') as f:
                print(f'sub_{code}_{task}: Not found and skipped', file=f)
            continue
    # loading and downsampling raw data to 1000hz
    # This is mainly so that the data is less noisy and takes up less space
    # Additional benefit is that data is now already in ms format
    # The try loop checks for correct referencing in the BrainVision .vhdr file
    try:
        raw = mne.io.read_raw_brainvision(path + f"/pp_{code}_{task}.vhdr")
        raw.load_data()
        raw_re = raw.copy().resample(sfreq=sample_rate)
    except FileNotFoundError:
        with open(output + 'log.txt', 'a') as f:
            print(f'pp_{code}_{task}: Could not find .vmrk file, check .vhdr file for correct reference!' file=f)
        continue

    # Cropping the file to trial length (15 Minutes), if the trial was shorter than
    # 15 Minutes no end point is set
        try:
            raw_crop = raw_re.crop(tmin=start,tmax=stop)
        except ValueError:
            raw_crop = raw_re.crop(tmin=start)
            with open(output + 'log.txt', 'a') as f:
                print(f'sub_{code}_{task}: Shorter than 900 seconds', file=f)
            
        
    # Notch filter at 50hz bandpassing at 0.05 - 150hz
        notch = raw_crop.notch_filter(freqs=50)
        filtered = notch.filter(l_freq=0.05, h_freq=150)
    # Bandfiltering the data and writing a new ecg file into an array
        not_used_Peaks,ecg_idx, average_pulse, ecg = mne.preprocessing.find_ecg_events(filtered, ch_name='ECG', event_id=1, qrs_threshold='auto', return_ecg=True)
        ecg = ecg.reshape(-1)
    # Inverting the ecg array because the polarity is upside down for some 
    # reason and would mislabel Q and S Peaks otherwise
        ecg = -ecg
    # Handle export to Heartpy for further analyses
        df = pd.DataFrame({'hr':ecg})
        df.to_csv('ecg.csv', index=False)
        data = hp.get_data('ecg.csv', column_name='hr')
        wd, m = hp.process(data, sample_rate)
        rpeaks = wd['peaklist']
        rejects = wd['removed_beats']
        rpeaks = [i for i in rpeaks if i not in rejects]
        IBI = np.diff(rpeaks)
    # Create .csv file with ms timings of peaks and rejected peaks, as well as BPM    
        df = pd.DataFrame({'R Peaks':rpeaks})
        df.loc[:,'Rejected'] = pd.Series(rejects)
        df.loc[:,'IBI'] = pd.Series(IBI)
        df['BPM'] = [None] * len(df)
        df.at[0,'BPM'] = round(m['bpm'])
    # Set-up for scrollable html file with raw ecg and markers for peaks and rejected peaks
        time = np.arange(len(ecg))
        time = time.reshape(-1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=ecg, mode='lines', name='ECG Signal'))
        fig.add_trace(go.Scatter(x=rpeaks, y=[ecg[i] for i in rpeaks], mode='markers', name='R-Peaks'))
        fig.add_trace(go.Scatter(x=rejects, y=[ecg[i] for i in rejects], mode='markers', name='Rejected Peaks'))
        fig.update_layout(title=f'Scrollable Line Graph for sub_{code}-{out}', xaxis_title='Time (ms)', yaxis_title='Volt')
    # Export the plot as a scrollable image
        pio.write_html(fig, file=output + f'graph_sub_{code}-{out}.html', auto_open=False, full_html=False)
    # Print dataframe with peaks and rejects into terminal to check
        print(df)
    # Output csv
        df.to_csv(output + f"sub_{code}_{out}.csv", sep=';', decimal=",", index=False)  # Depending on your location you might to invert the decimal and column separators
        print(f"Saving .csv for {out}-condition for sub_{code}")
    #Clean-up for temporary ecg export file
        os.remove('ecg.csv')
print("All data processed")
