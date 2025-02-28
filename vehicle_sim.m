function [next_state, reward, done] = step(obj, action)
    % Step function for RIS simulation with dynamic car movement
    % Inputs:
    %   obj: Simulation object with properties
    %   action: RIS phase configuration from the agent
    % Outputs:
    %   next_state: Next state observation
    %   reward: Reward for the action
    %   done: Boolean flag indicating if episode is terminated
    
    % Check if episode has already terminated
    if isEpisodeDone(obj) || obj.stepCount >= obj.maxSteps
        done = true;
        next_state = obj.getState();
        reward = 0;
        return;
    end
    
    % Apply RIS phase configuration from action
    ris_phases = action(1:obj.Nr);
    obj.phi = diag(exp(1i * 2 * pi * ris_phases));
    
    % Compute channel matrices based on current positions
    [obj.H_br, obj.H_bt, obj.H_rt] = computeChannelMatrices(obj);
    
    % Compute performance metrics
    [W, obj.FIM] = computeWx(obj);
    covar_matrix = W * W';
    [peb] = obj.calculatePerformanceMetrics(covar_matrix);
    
    % Calculate minimum data rate
    R_min = computeR_min(obj);
    
    % Update the vehicle's position
    if ~isempty(obj.destination) && ~isequal(obj.car_loc, obj.destination)
        % Calculate direction vector
        direction_vector = obj.destination - obj.car_loc;
        distance = norm(direction_vector);
        
        % Check if destination reached
        if distance < (obj.speed * obj.dt)
            % Generate new random destination within boundaries
            obj.destination = [
                randi([obj.boundary(1,1), obj.boundary(2,1)]), 
                randi([obj.boundary(1,2), obj.boundary(2,2)]), 
                0  % Keep Z at ground level
            ];
            
            % Debug output (can be disabled in production)
            if obj.verbose
                fprintf('Car reached destination. New destination: [%.1f, %.1f, %.1f]\n', ...
                    obj.destination(1), obj.destination(2), obj.destination(3));
            end
        else
            % Normalize direction vector and update position
            direction = direction_vector / distance;
            obj.car_loc = obj.car_loc + direction * obj.speed * obj.dt;
            
            % Debug output (can be disabled in production)
            if obj.verbose && mod(obj.stepCount, 100) == 0
                fprintf('Step %d: Car at [%.1f, %.1f, %.1f], PEB: %.6f\n', ...
                    obj.stepCount, obj.car_loc(1), obj.car_loc(2), obj.car_loc(3), peb);
            end
        end
    else
        % Initialize destination if not set
        if isempty(obj.destination)
            obj.destination = [
                randi([obj.boundary(1,1), obj.boundary(2,1)]), 
                randi([obj.boundary(1,2), obj.boundary(2,2)]), 
                0  % Keep Z at ground level
            ];
        end
    end
    
    % Update the target location (if the car is the dynamic target)
    obj.target_loc = obj.car_loc;
    
    % Update time and step counter
    obj.time = obj.time + obj.dt;
    obj.stepCount = obj.stepCount + 1;
    
    % Store trajectory data for visualization
    obj.trajectory(obj.stepCount,:) = obj.car_loc;
    obj.peb_history(obj.stepCount) = peb;
    obj.time_history(obj.stepCount) = obj.time;
    
    % Recompute geometric parameters based on new positions
    [obj.L1, obj.L2, obj.L3, obj.L_proj1, obj.L_proj2, obj.L_proj3, obj.delays, obj.angles] = ...
        computeGeometricParameters(obj);
    
    % Get the new state
    next_state = getState(obj);
    
    % Calculate reward
    reward = obj.computeReward(peb, obj.rate, R_min);
    
    % Check if the episode should be terminated
    done = isEpisodeDone(obj) || (obj.stepCount >= obj.maxSteps);
end

% Helper function to compute channel matrices
function [H_br, H_bt, H_rt] = computeChannelMatrices(obj)
    % Extract locations
    bs_loc = obj.bs_loc;
    ris_loc = obj.ris_loc;
    target_loc = obj.target_loc;
    
    % Calculate geometric parameters if not already calculated
    if isempty(obj.angles)
        [~, ~, ~, ~, ~, ~, ~, angles] = computeGeometricParameters(obj);
        obj.angles = angles;
    end
    
    % Parameters
    fc = obj.fc;           % Carrier frequency
    lambda = 3e8/fc;       % Wavelength
    Nr = obj.Nr;           % Number of RIS elements
    Nt = obj.Nt;           % Number of BS antennas
    
    % Calculate channel matrices using your existing channel model
    % This is a placeholder - replace with your actual implementation
    
    % BS to RIS channel (H_br)
    H_br = channelModel(obj, obj.angles.theta_br, obj.angles.phi_br, Nr, Nt, lambda, bs_loc, ris_loc);
    
    % BS to Target channel (H_bt)
    H_bt = channelModel(obj, obj.angles.theta_bt, obj.angles.phi_bt, 1, Nt, lambda, bs_loc, target_loc);
    
    % RIS to Target channel (H_rt)
    H_rt = channelModel(obj, obj.angles.theta_rt, obj.angles.phi_rt, 1, Nr, lambda, ris_loc, target_loc);
end

% Simplified channel model function (replace with your actual implementation)
function H = channelModel(obj, theta, phi, Nr, Nt, lambda, tx_loc, rx_loc)
    % This function should be replaced with your actual channel model
    
    % Calculate distance
    d = norm(tx_loc - rx_loc);
    
    % Path loss
    path_loss = lambda/(4*pi*d);
    
    % Array steering vectors
    a_r = arraySteering(Nr, theta, phi, lambda);
    a_t = arraySteering(Nt, theta, phi, lambda);
    
    % LOS component
    H_los = a_r * a_t';
    
    % Add fading component (if used in your model)
    if obj.use_fading
        H_fading = (randn(Nr, Nt) + 1i*randn(Nr, Nt))/sqrt(2) * obj.fading_factor;
        H = path_loss * (H_los + H_fading);
    else
        H = path_loss * H_los;
    end
end

% Function to determine if episode is done
function done = isEpisodeDone(obj)
    % Check if car is out of boundaries
    if any(obj.car_loc < obj.boundary(1,:)) || any(obj.car_loc > obj.boundary(2,:))
        done = true;
        return;
    end
    
    % Check if maximum steps reached
    if obj.stepCount >= obj.maxSteps
        done = true;
        return;
    end
    
    % Episode is not done
    done = false;
end

% Function to get current state
function state = getState(obj)
    % Create state vector based on:
    % 1. Car's position
    % 2. Geometric parameters (angles, distances)
    % 3. Current RIS configuration
    
    % Extract diagonal of phi matrix
    phi_diag = diag(obj.phi);
    phi_angles = angle(phi_diag) / (2*pi);
    
    % Combine all state components
    state = [
        obj.car_loc(:)', ...                        % Car position [x, y, z]
        obj.L1, obj.L2, obj.L3, ...                 % Path lengths
        obj.angles.theta_bt, obj.angles.phi_bt, ... % BS-Target angles
        obj.angles.theta_br, obj.angles.phi_br, ... % BS-RIS angles
        obj.angles.theta_rt, obj.angles.phi_rt, ... % RIS-Target angles
        phi_angles(:)'                              % Current RIS configuration
    ];
end

% Function to compute geometric parameters
function [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj)
    % Extract locations
    bs_loc = obj.bs_loc;
    ris_loc = obj.ris_loc;
    target_loc = obj.target_loc;
    
    % Direct path: BS to Target
    L1 = norm(bs_loc - target_loc);
    
    % First reflection path: BS to RIS
    L2 = norm(bs_loc - ris_loc);
    
    % Second reflection path: RIS to Target
    L3 = norm(ris_loc - target_loc);
    
    % Projected distances (horizontal plane)
    L_proj1 = norm(bs_loc(1:2) - target_loc(1:2));
    L_proj2 = norm(bs_loc(1:2) - ris_loc(1:2));
    L_proj3 = norm(ris_loc(1:2) - target_loc(1:2));
    
    % Calculate delays
    c = 3e8;  % Speed of light (m/s)
    delays = [L1/c, (L2+L3)/c];
    
    % Calculate angles
    % Elevation angles (with respect to horizontal plane)
    theta_bt = atan2(target_loc(3) - bs_loc(3), L_proj1);
    theta_br = atan2(ris_loc(3) - bs_loc(3), L_proj2);
    theta_rt = atan2(target_loc(3) - ris_loc(3), L_proj3);
    
    % Azimuth angles
    phi_bt = atan2(target_loc(2) - bs_loc(2), target_loc(1) - bs_loc(1));
    phi_br = atan2(ris_loc(2) - bs_loc(2), ris_loc(1) - bs_loc(1));
    phi_rt = atan2(target_loc(2) - ris_loc(2), target_loc(1) - ris_loc(1));
    
    % Package angles
    angles = struct('theta_bt', theta_bt, 'theta_br', theta_br, 'theta_rt', theta_rt, ...
                   'phi_bt', phi_bt, 'phi_br', phi_br, 'phi_rt', phi_rt);
end

% Function to create array steering vector
function a = arraySteering(N, theta, phi, lambda)
    % Element spacing
    d = lambda/2;
    
    % Initialize steering vector
    a = zeros(N, 1);
    
    % Calculate steering vector elements
    for n = 0:N-1
        a(n+1) = exp(1i * 2 * pi * d * n * sin(theta) * cos(phi) / lambda);
    end
end

% Function to compute minimum data rate
function R_min = computeR_min(obj)
    % This is a placeholder - replace with your actual implementation
    % Calculate the minimum achievable data rate based on channel conditions
    
    % Extract channels
    H_bt = obj.H_bt;
    H_br = obj.H_br;
    H_rt = obj.H_rt;
    phi = obj.phi;
    
    % Calculate effective channel
    H_eff = H_bt + H_br * phi * H_rt;
    
    % Calculate SNR (simplified)
    snr_linear = 10^(obj.snr_db/10);
    
    % Calculate rate (bits/s/Hz)
    R_min = log2(1 + snr_linear * norm(H_eff, 'fro')^2);
end

% Function to compute beamformer matrix Wx
function [W, FIM] = computeWx(obj)
    % This is a placeholder - replace with your actual implementation
    % Compute the beamformer matrix and Fisher Information Matrix
    
    % Extract channels
    H_bt = obj.H_bt;
    H_br = obj.H_br;
    H_rt = obj.H_rt;
    phi = obj.phi;
    
    % Calculate effective channel
    H_eff = H_bt + H_br * phi * H_rt;
    
    % Calculate W (simplified)
    W = H_eff' / (norm(H_eff, 'fro') + eps);
    
    % Calculate FIM (simplified)
    snr_linear = 10^(obj.snr_db/10);
    FIM = snr_linear * (W * W');
end