clear;

m = [1, 2, 3; 4, 5, 6; 7, 8, 9];
m_pad = zeros(5,5);
m_pad(2:4,2:4) = m;
m_smooth = [];

temp = [];
[r, s] = size(m_pad);

% smooth then normalize
for i =2:4
    for j = 2:4
        temp(1) = m_pad(i-1, j-1);
        temp(2) = m_pad(i-1, j);
        temp(3) = m_pad(i-1, j+1);
        temp(4) = m_pad(i, j-1);
        temp(5) = m_pad(i,j);
        temp(6) = m_pad(i,j+1);
        temp(7) = m_pad(i+1,j-1);
        temp(8) = m_pad(i+1,j);
        temp(9) = m_pad(i+1,j+1);
        m_smooth(i-1,j-1) = (sum(temp))/9;
    end
end


for i = 1:3
    for j = 1:3
         smooth_then_norm (i,j) = val_norm(m_smooth(i,j), min(m_smooth(:)), max(m_smooth(:)));
    end
end


m = [1, 2, 3; 4, 5, 6; 7, 8, 9];
m_pad = zeros(5,5);
m_pad(2:4,2:4) = m;


temp = [];
[r, s] = size(m_pad);

%normalize then smooth
for i =2:4
    for j = 2:4
        temp(1) = val_norm(m_pad(i-1, j-1), 1, 9);
        temp(2) = val_norm(m_pad(i-1, j), 1, 9);
        temp(3) = val_norm(m_pad(i-1, j+1), 1, 9);
        temp(4) = val_norm(m_pad(i, j-1), 1, 9);
        temp(5) = val_norm(m_pad(i,j), 1, 9);
        temp(6) = val_norm(m_pad(i,j+1), 1, 9);
        temp(7) = val_norm(m_pad(i+1,j-1), 1, 9);
        temp(8) = val_norm(m_pad(i+1,j), 1, 9);
        temp(9) = val_norm(m_pad(i+1,j+1), 1, 9);
        norm_to_smooth(i-1,j-1) = (sum(temp))/9;
    end
end

%% for main.m if I feel like it
% Smooth the filtered images
for k = 1:nscale
    for l = 1:norient
        for i = 1:rA
            for j = 1:sA
                switch_expression = str2double(strcat(num2str(i),num2str(j)));
                switch switch_expression
                    case 11 

                end
            end
        end
    end
end
