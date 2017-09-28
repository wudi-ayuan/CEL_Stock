function [ Count, Counting ] = eval_perform( Count, final, Test_Label, y_hat, t, Counting, No_E)
%EVAL_PERFORM Summary of this function goes here
%   Detailed explanation goes here
        %% Performance Analysis
        % For PER
        if (Test_Label(t) ~= final(t))
            Count(1) = Count(1) + 1;
        end
        % For FAR
        if ((Test_Label(t) == 2) && (final(t) == 1))
            Count(2) = Count(2) + 1;
        end
        % For MDR
        if ((Test_Label(t) == 1) && (final(t) == 2))
            Count(3) = Count(3) + 1;
        end
        if (Test_Label(t) == 2)
            % For FAR
            Count(4) = Count(4) + 1;
        else
            % For MDR
            Count(5) = Count(5) + 1;
        end
      
        %% Performance Analysis for Each Expert
        for e = 1:No_E
            % For PER
            if (Test_Label(t) ~= y_hat(t,e))
                Counting(1,e) = Counting(1,e) + 1;
            end
            % For FAR
            if ((Test_Label(t) == 2) && ( y_hat(t,e) == 1))
                Counting(2,e) = Counting(2,e) + 1;
            end
            % For MDR
            if ((Test_Label(t) == 1) && ( y_hat(t,e) == 2))
                Counting(3,e) = Counting(3,e) + 1;
            end
            if (Test_Label(t) == 2)
                % For FAR
                Counting(4,e) = Counting(4,e) + 1;
            else
                % For MDR
                Counting(5,e) = Counting(5,e) + 1;
            end
        end

end

