function data_out = extract_stats(data)

%the 11 statistical functionals are
%mean, std-dev, skewness, kurtosis, range, min, max, first quantile, third
%quantile, median quantile, inter-quantile range


nframes = size(data,1);
ncols = size(data,2);
% data_out = zeros(floor(nframes/move_gap),ncols);
data_mean = [];
data_stddev = [];
data_skew = [];
data_kurt = [];
data_max = [];
data_min = [];
data_range = [];
data_median = [];
data_q1 = [];
data_q3 = [];
data_iqrange = [];


   counter = 1;
    current = data;
    
    data_mean(counter,:) = mean(current);
    data_stddev(counter,:) = std(current);
    data_skew(counter,:) = skewness(current);
    data_kurt(counter,:) = kurtosis(current);
    data_max(counter,:) = max(current);
    data_min(counter,:) = min(current);
    data_range(counter,:) = data_max(counter,:) - data_min(counter,:);
    data_median(counter,:) = median(current);

    for col_ind = 1:ncols
        current_feat = current(:,col_ind);
        data_q1(counter,col_ind) = median(current_feat(find(current_feat<median(current_feat))));
        data_q3(counter,col_ind) = median(current_feat(find(current_feat>median(current_feat))));
        data_iqrange(counter,col_ind) = data_q3(counter,col_ind) - data_q1(counter,col_ind);
    end

data_out= [data_mean,data_stddev,data_skew,data_kurt,data_max,data_min,data_range,data_median,data_q1,data_q3,data_iqrange];
