function ref = getRefSep(q)
    Ndem = length(q);
    ref_cell = cell(1,Ndem);
    for idx=1:Ndem
        ref_cell{idx} = idx.*ones(1,length(q{idx}));
    end
    ref = cell2mat(ref_cell);
end