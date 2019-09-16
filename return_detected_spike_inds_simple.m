function [SpikeInds, SpikeAmps, SpikePeaks] = return_detected_spike_inds_simple(D,Thresh,DT,Tb,Ta,SR);
% Get the (processed) signal and return the indices where spikes are found.
%
% SR is sampling rate in Hz
% DT is dead time in seconds
% Tb and Ta are the times before and after the spike peak to be extracted in ms
%
% NOTE: this is a simplified version, with fat removed. YBS jan 2014

Si = 1/(SR/1000); % sampling interval (ms)
Sb = round(Tb/Si); % samples before
Sa = round(Ta/Si); % samples after

SpikeInds = [];
SpikeAmps = [];
SpikePeaks = [];

% Find the peaks - which are defined as these points where the slope before and slope after are different sign
Slope_Sign  = sign(diff(D));

% Find peaks based on first second derivative - this specifies an upslope
% followed by a downslope or a plateu actualy
Pos_Peaks = find( (Slope_Sign(1:end-1) + Slope_Sign(2:end)) <  2   & Slope_Sign(1:end-1) > 0);
Neg_Peaks = find( (Slope_Sign(1:end-1) + Slope_Sign(2:end)) > -2   & Slope_Sign(1:end-1) < 0);
Pos_Peaks = Pos_Peaks + 1;
Neg_Peaks = Neg_Peaks + 1;

% Take only peaks above threshhold
PosSpikeInds = intersect(Pos_Peaks,find(D > Thresh));
NegSpikeInds = intersect(Neg_Peaks,find(D < -Thresh));

% combine the positive and negative peaks
SpikeInds = union(PosSpikeInds,NegSpikeInds);

% Apply dead time for negative and positive, take largest spike
[SpikeInds, Kept_Spikes] = apply_dead_time(SpikeInds,'takeL',DT,Si,D);

% Retain spikes fully contained within the signal
SpikeInds = SpikeInds((SpikeInds > Sb) & (SpikeInds < (length(D) - Sa)));

for i = 1:length(SpikeInds)
    thisSpike =    (D([SpikeInds(i)-Sb:SpikeInds(i)+Sa]));
    SpikeAmps(i) = range(thisSpike);
end
    
SpikePeaks = D(SpikeInds);

return

