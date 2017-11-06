function ref = getRefSepDiff(q)
    Ndem = length(q);
    ref_cell = cell(1,Ndem);
    signal = 1;
    for idx=1:Ndem
        ref_cell{idx} = signal.*ones(1,length(q{idx}));
        signal = -signal;
    end
    ref = cell2mat(ref_cell);
end