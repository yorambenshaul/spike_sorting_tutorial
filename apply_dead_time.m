function [SpikeInds Kept_Spikes] = apply_dead_time(SpikeInds,select_criterion,DT,Si,D)
% Function extracted from the detect spikes function, which has been now
% parsed.

% SpikeInds are the indices of detected spikes, in samples.
% select_criterion can be either of these - 
% 'takeL'  take largest
% 'takeLP' take largest positive
% 'takeLN' take largest negative
% 'takeF' take first
% 'takeLst' take last
% DT is dead time in seconds
% Si is sampling interval in ms
% D is the data, which is required for selection criteria based on signal
% magnitude

% Get intervals between detected spikes:
Spike_Intervals = diff(SpikeInds);
% Find which are the dense spikes
DSi    = find(Spike_Intervals*Si < DT);

% Go over all cases of dense spikes
dind = 1;   Done = 0;
if ~isempty(DSi)
    while  ~Done
        k = 1;
        not_this_done = 1;
        % Dspikes includes at the end of this stage one full set of dense spikes
        Dspikes = [];
        Dspikes(k) = DSi(dind);
        % Complete this group
        while dind < length(DSi) && not_this_done            
            dind = dind + 1;        
            % We check if the next close interval follows immediately
            if (DSi(dind) - Dspikes(k)) == 1
                % If it does we add it to the Dspikes
                k = k + 1;   
                Dspikes(k) = DSi(dind);
            else
                % And if it does not, we are finished with the current group
                not_this_done = 0;             
            end
        end
        % This is the exit condition - if the last spike interval in current group is the last one
        % we exit.
        if Dspikes(k) == DSi(end)
            Done = 1;
        end
        % Now for each group
        % Get the spike indices form the interval indices:
        these_spike_inds = [Dspikes Dspikes(end)+1];
        % Now get the 
             
        % Based on the selection criterion - select one spike
        switch select_criterion
            case 'takeL'  
            these_D_vals     = D(SpikeInds(these_spike_inds));                
            [tmp good_one] = max(abs(these_D_vals));
            case 'takeLP'
            these_D_vals     = D(SpikeInds(these_spike_inds));                
            [tmp good_one] = max(these_D_vals);
            case 'takeLN'
            these_D_vals     = D(SpikeInds(these_spike_inds));                
            [tmp good_one] = min(these_D_vals);
            case 'takeF'
            good_one = 1;
            case 'takeLst'
            good_one = length(these_spike_inds);
        end
            
        % Build the group of spikes to exclude
        exclude = setdiff(these_spike_inds,these_spike_inds(good_one));
        % And exclude them from the set of detected spikes. 
        SpikeInds(exclude) = -1;                 
    end
end

Kept_Spikes = find(SpikeInds > 0);
SpikeInds   = SpikeInds(Kept_Spikes);

return
