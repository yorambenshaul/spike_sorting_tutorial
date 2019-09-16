% simulate data for class 14


clear
close all

%% generate the data

% First we generate the spike shapes
% load prep-prepared spike shapes
load('example_pcs'); % make sure they are on the path
% generate 3 basic spike shapes - using lnera combinations of the PCs
SpikeShape{1} =  -3*example_pcs(:,1) + 1 * example_pcs(:,2);
SpikeShape{2} =  3*example_pcs(:,1) - 3 * example_pcs(:,2);
SpikeShape{3} =  5*example_pcs(:,1) + 0.05 * example_pcs(:,2);
% generate a new figure
figure
plot(SpikeShape{1},'r')
hold on
plot(SpikeShape{2},'g')
plot(SpikeShape{3},'b')
legend({'S1','S2','S3'});


% general parameters related to the data
SR = 24000;     % Sampling rate - in Hz
refract_ms = 15; % refractory period in ms
ref_samps = SR * refract_ms/1000; % refractory period in samples


%% Generate the "behavioral task"

% Describe trial sequence  - we have 8 different behavioral conditions,
% 1:8, and 0 between them
% trial sequence - describing stims and times
one_block = [0 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8];
% trial sequence
Nr = 12; % number of repeats
ts =  repmat(one_block,1,Nr); % the entire sequence of trials - we could also do a ranperms on this, if we want to change the order in each sequence
ts = [ts 0]; % we need a little extra at the end of every trial
% trial time in seconds
T = 1; % time for each epoch (inter trial interval and the stimulus)
tt =  T * [1:length(ts)];


% define spike rates for the 8 behvioral states
% This is for 3 neurons that we defined above
Base_rates = [10 5 5]; 
Stim_rates{1} = [10   10  20 30 50 30  20 10 ];
Stim_rates{2} = [40   5   5  5  5  5   5  5];
Stim_rates{3} = [5    30  5  30 5  30  5  30];

%% create data vectors for the trial sequence
% initialize the data vectors
for si = 1:length(SpikeShape)
    data{si} = [];
end    
% generate vectors of spike times
for si = 1:length(SpikeShape) % over the different neurons
    for i = 1:length(ts)      % over all epochs in the session 
        % if the epoch is an ITI use the baseline rates
        if ts(i) == 0
            this_rate = Base_rates(si);
        % otherwise use the rate for this specific epoch
        else
            % This line means: take the firing rates for the condition
            % specific in ts
            this_rate = Stim_rates{si}(ts(i));
        end
        % now construct a sequence of spike times. 
        % we continuously increase this vector (for this unit) and we
        % generate a vector of 1 and 0s, whose probability is the
        % probability of a spike within one sampling interval (1 divided by
        % the SR).
        % We build a vecor thta is T seconds long
        data{si} = [data{si} binornd(1,this_rate/SR,1,SR*T)];
    end
end


% Plot the ISI histograms

% apply a refractory period
% the loop looks for all close intervals and deletes them
% as necessary until no intervals smaller than the refractory period
% are present.
for si = 1:length(SpikeShape)
    sp_times{si} = find(data{si});
    % as long as the minimal difference is small than the samples in
    % the refractory period ....
    while min(diff(sp_times{si})) < ref_samps
        for i = 2:length(sp_times{si})
            % delete the first spike in an interval that you find that is too small
            if sp_times{si}(i) - sp_times{si}(i-1) < ref_samps
                sp_times{si}(i) = [];
                break
            end
        end
    end
end

        
% apply a refractory period
% the loop looks for all close intervals and deletes them
% as necessary until no intervals smaller than the refractory period
% are present.
for si = 1:length(SpikeShape)
    disp(['removing dense spikes from class ' num2str(si) ])
    sp_times{si} = find(data{si});
    original_sp_times{si} = sp_times{si};
    % as long as the minimal difference is small than the samples in
    % the refractory period ....
    while min(diff(sp_times{si})) < ref_samps
        for i = 2:length(sp_times{si})
            % delete the first spike in an interval that you find that is too small
            if sp_times{si}(i) - sp_times{si}(i-1) < ref_samps
                sp_times{si}(i) = [];
                break
            end
        end
    end
end



% Plot the ISI histograms
figure
for si = 1:length(SpikeShape)
    sp_ints = diff(original_sp_times{si});
    % convert to ms
    sp_ints = 1000* sp_ints/SR;
    subplot(length(SpikeShape),1,si)
    hist(sp_ints,[0:5:300]);
    xlabel('ms')
    set(gca,'xlim',[0 300])
    title(['ISIs for class ' num2str(si) ' before removing dense spikes'])
end

figure
for si = 1:length(SpikeShape)
    sp_ints = diff(sp_times{si});
    % convert to ms
    sp_ints = 1000* sp_ints/SR;
    subplot(length(SpikeShape),1,si)
    hist(sp_ints,[0:5:300]);
    xlabel('ms')
    set(gca,'xlim',[0 300])
    title(['ISIs for class ' num2str(si) ' after removing dense spikes'])
end



%% next, we want to convert our vector of spike times to a vector of spikes shapes
% Reset the data vector since we will soon replace spike times
% with actual spike shapes
for si = 1:length(SpikeShape)
    data{si} = 0*data{si};
end 
% now add spikes to the signals
SL = length(SpikeShape{1}); 
for si = 1:length(SpikeShape) % for each neuron
    for i = 1:length(sp_times{si}) % for each spike time
        if sp_times{si}(i)+SL < length(data{si}) % if the spike will fit in this time points - add the spike shape to the data
            data{si}([sp_times{si}(i):sp_times{si}(i)+SL-1]) = SpikeShape{si};
        end
    end
end

% Plot the individual spikes
figure
timevec = [1:length(data{si})] /SR;
for si = 1:length(SpikeShape)   
    sh(si) = subplot(length(SpikeShape),1,si);
    plot(timevec,data{si})
    xlabel('s')
    set(gca,'xlim',[0 max(timevec)]);
    title(['clean spike vector for ' num2str(si) ])
end
linkaxes(sh,'x'); % This will force all the x axes to have the same scale if you change one of them
%set(get(gcf,'children'),'xlim',[0 1])
 

% Now add the data together and then add the noise
all_data = data{1};
for si = 2:length(SpikeShape)   
    all_data = all_data + data{si};    
end

% Generate the noise vector
Wnoise = 0.05*randn(size(all_data)); % we generate a vector of white noise
% Truncate it to 0.1
Wnoise(abs(Wnoise)>0.1) = 0.1;
all_data_with_noise = all_data + Wnoise;

% Add a 50Hz noise vector 
dt = 1/SR; % the sampling interval
t= [0:dt:ceil(length(all_data_with_noise)/SR)]; % a time vector for the sine wave
C50   = sin(2*pi*50*t); % generate a sine wave with a frequency of 50Hz
C50   = C50(1:length(all_data)); % truncate it if needed to fir the data vector length
all_data_with_noise_and_line = all_data_with_noise + C50; % add the 50Hz noise

%% now we want to filter the data as well
% We want to filter at 300HZ
% define a butterworth filter with a 50Hz high pass. 7 poles, because it
% works
[Bh,Ah] = butter(7,0.025,'high');
% fvtool(Bh,Ah); % uncomment this to see the filter repsonse proprties and
% other aspects
filtered_data = filtfilt(Bh,Ah,all_data_with_noise_and_line); % filter the data...


% show the data before and after fitering 
figure
sh2(1) = subplot(4,1,1);
plot(timevec,all_data)
set(gca,'xlim',[0 max(timevec)])
title('data before adding noise')
sh2(2) = subplot(4,1,2);
plot(timevec,all_data_with_noise)
set(gca,'xlim',[0 max(timevec)])
title('data with white noise')
sh2(3) = subplot(4,1,3);
plot(timevec,all_data_with_noise_and_line)
set(gca,'xlim',[0 max(timevec)])
title('data with white and line noise')
sh2(4) = subplot(4,1,4);
plot(timevec,filtered_data)
set(gca,'xlim',[0 max(timevec)])
title('filtered data')
linkaxes(sh2)
%set(get(gcf,'children'),'xlim',[0.2 0.4],'ylim',[-2.5 2.5])


save('neuronal_sim_data');
clear
load('neuronal_sim_data');


%% now we are starting to work back from the data and extract the information

% Plot  histogram of amplitudes, this is relevant for the spike detection 
% we can use the distribution to decide on the spike threshold for
% detection. here we set a noise of 0.1 max and the spikes are a bit more than 1, so 1 is a good thresold
% the alterantive is to look at the entire data in the time domain 
sh31 = subplot(2,1,1);
histogram(filtered_data,[-2:0.01:2])
%hist(filtered_data,[-2:0.01:2])
title('normal y scale')
sh32 = subplot(2,1,2);
%hist(filtered_data,[-2:0.01:2])
histogram(filtered_data,[-2:0.01:2])
set(gca,'yScale','log')
linkaxes([sh31,sh32],'x')
title('log y scale');


%% detect the spikes

% This may be a good time to talk about the problems of multiple spike
% detections, about alligning spikes, and about tetrode and multichannel
% recordings

thresh = 1; % define the threshols
% This is another option ... which will not work without some scaling
% thresh = thselect(filtered_data,'minimaxi'); % do "doc thselect" to learn more
% about this
%[SpikeInds SpikeAmps SpikePeaks] = return_detected_spike_inds_cleaned(filtered_data,0.5,5,2,2,SR);
[SpikeInds SpikeAmps SpikePeaks] = return_detected_spike_inds_simple(filtered_data,thresh,5,2,2,SR);

% plot the detected spikes
figure
plot(timevec,filtered_data)
hold on
plot(timevec(SpikeInds),filtered_data(SpikeInds),'r.')
title([' Threshold: ' num2str(thresh)])


%% extract the spikes
% spikes are defined as 30 samples before and 40 samples after
% so a spike is almost 2 ms long
% extract the spikes:
Sb = 30;
Sa = 40;
for i = 1:length(SpikeInds)
    SpikeMat(i,:) = filtered_data(SpikeInds(i)-Sb:SpikeInds(i)+Sa);
end
% it is possible to do spike alligning, but probably a waste of time
    

%%  calculate PCs and projections
% this gives us the PCs, the projections on the PCs, and variance explained
% by the PCs
[coeff, score, latent] = pca(SpikeMat);

%% cluster the spikes based on the projections
% Cluster the spikes based on first two PCs
nclust = 3;
IDX = kmeans(score(:,1:2),nclust,'Start','cluster','Replicates',5);
uinds = unique(IDX);

% Plot the spikes and Projections before and after sorting
figure
subplot(3,1,1)
plot(SpikeMat')
title('Spikes before sorting')
subplot(3,1,2)
co = 'rgbkm';
for i = 1:nclust
    %subplot(nclust,1,i)
    these = find(IDX == i);
    these_spikes = SpikeMat(these,:);    
    ph = plot(these_spikes');
    set(ph,'color',co(i));
    hold on    
    title(['class ' num2str(i) , ' N = ' num2str(length(these))])
end
title('Spikes after sorting')
subplot(3,1,3)
co = 'rgbkm';
for i = 1:nclust
    %subplot(nclust,1,i)
    these = find(IDX == i);
    these_spikes = SpikeMat(these,:);
    mean_these = mean(these_spikes);
    sd_these = std(these_spikes);
    ph = plot(mean_these);
    set(ph,'color',co(i));
    hold on
    ph = plot(mean_these+sd_these,':');
    set(ph,'color',co(i));
    ph = plot(mean_these-sd_these,':');
    set(ph,'color',co(i));
    title(['class ' num2str(i) , ' N = ' num2str(length(these))])
end
title('Mean and SD of spikes after sorting')
set(get(gcf,'children'),'xlim',[0 size(SpikeMat,2)])


%% plot the projections on the PCs 
figure
subplot(1,2,1)
plot(score(:,1),score(:,2),'k.')
title('projections of spikes before sorting')
axis equal
xlabel('pc1')
ylabel('pc2')
% And after sorting
subplot(1,2,2)
co = 'rgbkm';
for i = 1:nclust
    these = find(IDX == i);
    ph = plot(score(these,1),score(these,2),'.');
    set(ph,'color',co(i));
    hold on
end
axis equal
xlabel('pc1')
ylabel('pc2')
title('projections of spikes after sorting')


% plot the variance explained by the first 10 PCs and the PCs themseleves
figure
subplot(1,2,1)
bar(latent(1:10))
title('variances')
subplot(1,2,2)
plot(coeff(:,1:3))
legend({'1','2','3'})
title('PCs')



%% analyze the data - now we generate the vector of spike times for the neurons that we constructed initially
% generate a vector of spike times for each f the clusters - hopefully
% these are the original neurons that we had
for i = 1:nclust
    these = find(IDX == i);
    spike_times{i} = timevec(SpikeInds(these));    
end


% Now count how many spikes for each class in each epoch
ue = unique(ts); % these are the different epoch defined 
for ei = 1:length(ue) % go over each epoch
    % inds for this event
    epoch_inds  = find(ts == ue(ei)); % find those epochs in the trial sequenes
    epoch_times = tt(epoch_inds);     % These are the corresponding times of those trials
    for si = 1:nclust % for each of the clusters
        clear trial_spikes n_spikes full_trial_spikes;
        for ti = 1:length(epoch_inds) % trial indices            
            % find those spikes which are within the interval .... the time
            % we have is the end of the interval
            tmp_spike_inds = intersect(find(spike_times{si}< epoch_times(ti)),find(spike_times{si} >= epoch_times(ti)-T));                        
            trial_spikes{ti} = spike_times{si}(tmp_spike_inds) - (epoch_times(ti)-T);             
            n_spikes(ti) = length(trial_spikes{ti}); % count how many spikes within the interval
            if ei > 1 % if it is not a baseline epochs
                % save al spiketme. i take also some time before and after
                % the trial duration
                full_trial_spike_inds = intersect(find(spike_times{si}< epoch_times(ti)+0.3),find(spike_times{si} >= epoch_times(ti)-1.5*T));
                full_trial_spikes{ti} = spike_times{si}(full_trial_spike_inds) - (epoch_times(ti)-1.5*T);
            else
                full_trial_spikes{ti} = []; % we don't care about spikes in the baseline epoch
            end
        end
        % derive the values for each epoch and each spike cluster
        all_spikes_with_pre{si,ei}  = full_trial_spikes;
        all_spikes{si,ei}  = trial_spikes;
        all_counts{si,ei}  = n_spikes;
        mean_counts(si,ei) = mean(n_spikes); 
        se_counts(si,ei) = std(n_spikes)/sqrt(length(n_spikes));
    end
end
    


% make PSTHs of the data
for si = 1:nclust % over spike
    for ei = 1:max(ue) 
        these_trials = all_spikes_with_pre{si,ei+1}; % the first one is the baseline
        ntrials = length(these_trials);
        all_trial_spikes = [];
        % we construct a vector of spike times, where we just append the
        % trials one after the other
        for ti = 1:ntrials
            all_trial_spikes = [all_trial_spikes these_trials{ti}];
        end
        % build a hist vector, here I hard coded the trial duration, ideally this
        % should be specified based on the trial duration parameters
        timevec = [0:0.2:1.8];
        [psth{si,ei}  bincens ] = hist(all_trial_spikes,timevec);
    end
end


% do the statistics of the comparisons based on spike counts
% this is a Kurskal wallis test for the comparison of spike counts
clear stimP
for si = 1:nclust   
    baseline = all_counts{si,1};
    basegroup = zeros(1,length(baseline));
    for ei = 1:max(ue)              
        stim_counts = all_counts{si,ei+1};
        stim_group = ones(1,length(stim_counts));
        
        X = [baseline stim_counts];
        GROUP = [basegroup stim_group];        
        stimP(si,ei) = kruskalwallis(X,GROUP,'off');                       
    end
end

% define a signifiance threshold
Pthresh = 0.01;
sigstims = stimP < Pthresh; % for those epochs that are significantly different from baseline
    

% plot the PSTHs and raster displays
for si = 1:nclust % for each class    
    figure
    set(gcf,'name',['spike class ' num2str(si)])
    maxy = 0;
    for ei = 1:max(ue) % plot PSTHS for each of the epochs
        sh1(ei) =  subplot(2,8,ei);
        ph = plot(bincens,psth{si,ei});
        set(ph,'color','k','linewidth',3)
        maxy = max(maxy,max(get(gca,'ylim')));
        if ei > 1
            set(gca,'ytick',[])
        end
        th = title(['stim ' num2str(ei)]);
        
        % if significant make the axis red
        if sigstims(si,ei)
            set(gca,'color','r')
        end        
        if ei == 1 % put a label for the leftmost plot
            ylabel('rate Hz')
        end
    end    
    set(sh1,'ylim',[0 maxy],'xlim',[0 1.8],'xtick',[0:0.5:1.5])
    
    
    for ei = 1:max(ue)
        sh2(ei) = subplot(2,max(ue),max(ue)+ei);
        these_trials = all_spikes_with_pre{si,ei+1}; % the first one is the baseline
            
        tmp_spike_trials = [];
        tmp_spike_times = [];
        % below is a way to efficiently plot the raster display
        for ti = 1:ntrials % take the times within each trial
            tmp_spike_times =  [tmp_spike_times ; these_trials{ti}'];
            tmp_spike_trials = [tmp_spike_trials (ntrials+1-ti)*ones(1,length(these_trials{ti}))];
        end
        disp(ei);
        clear Y X
        % This is a trick to plot the spike times as a single line plot
        if ~isempty(tmp_spike_trials)
            Y(1,:) = tmp_spike_trials-0.3;
            Y(2,:) = tmp_spike_trials+0.3;
            Y(3,:) = nan(1,size(Y,2));
            X(1,:) = tmp_spike_times;
            X(2,:) = tmp_spike_times;
            X(3,:) = nan(1,size(X,2));
            nX = reshape(X,1,numel(X));
            nY = reshape(Y,1,numel(Y));
            ph = plot(nX,nY,'k');
        end
        if ei == 1
            ylabel('trial #')
        end
        xlabel('time (s)')
        % if significant amke the axis red
        if sigstims(si,ei)
            set(gca,'color','r')
        end
    end
    set(sh2,'ylim',[0 13],'xlim',[0 1.8],'xtick',[0:0.5:1.5],'ytick',[])
    
end


% plot the mean counts per epoch - here too I am being lazy and using 
% known values rather than parameters for the number of epochs
figure
plot(mean_counts(:,2:9)');
legend({'Class1','Class 2','Class 3'})
hold on
plot([mean_counts(:,2:9)+se_counts(:,2:9)]',':');
plot([mean_counts(:,2:9)-se_counts(:,2:9)]',':');
xlabel('stim %')
ylabel('Hz')
title('mean and standard error of the responses')


return


    
    
    
    
    


