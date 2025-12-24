sat = 'CL2';
start_date = 'Dec-09-2008';
end_date = 'Mar-1-2020';



if exist('AllModeData.mat') ~= 2
    TM = getTM(sat, {'HA300AL', 'HA300BL'}, [start_date, ' 00:00:00'], [end_date, ' 00:00:00']);
    save('AllModeData.mat', 'TM');
else
    load('AllModeData.mat');
end

cell_array = {};

for count_value = 2:length(TM.HA300BL)
    this_time = TM.time(count_value);
    this_value = TM.HA300BL(count_value);
    prior_value = TM.HA300BL(count_value-1);
    
    
    if this_value ~= 0 && this_value~=prior_value
        find_last_index = find(TM.HA300BL((count_value + 1):end) ~= this_value, 1) + count_value;
        event_end_time = TM.time(find_last_index);
        
        if this_value == 1
        	this_event = 'UNLOAD_MAN';
        elseif this_value == 2 && TM.HA300AL(count_value) == 1
            this_event = 'EAST_MAN';
        elseif this_value ==2 && TM.HA300AL(count_value) ==2
            this_event = 'WEST_MAN';
        elseif this_value ==3 && TM.HA300AL(count_value) ==3
            this_event = 'NORTH_MAN';
        elseif this_value ==3 && TM.HA300AL(count_value) ==4
            this_event = 'SOUTH_MAN';
        else
            display(['Value:', num2str(this_value), ' Direction:', num2str(TM.HA300AL(count_value)), '. Not valid.']); 
        end
        cell_array = [cell_array; {datestr(this_time),datestr(event_end_time),this_event}];
    end
    
end


 fid = fopen('CIEL2_Events.csv','wt');
 if fid>0
     for k=1:size(cell_array,1)
         row = sprintf('%s,',cell_array{k,:});
         row = [row(1:end-1)];
         fprintf(fid, [row,'\n']);
     end
     fclose(fid);
 end
