% System Parameters
Nb = 8;         % Number of BS antennas (ULA)
Nt = 4;         % Number of Vehicle antennas
Nx = 8;         % Number of RIS elements in x-direction
Ny = 8;         % Number of RIS elements in y-direction
Nr = Nx * Ny;   % Total number of RIS elements
Mb = 64;        % Number of OFDM subcarriers
lambda = 0.1;   % Wavelength (assuming frequency around 3GHz)
d = lambda/2;   % Inter-element spacing

% Initialize parameters
x_size = 1000;  % Environment size in meters
y_size = 1000;
z_size = 100;   % Added height dimension
dt = 0.1;       % Time step in seconds
speed = 10;     % Vehicle speed in m/s

% Initial positions
vehicle_pos = [500, 500, 0];  % Vehicle starts at ground level
goal_pos = [1000, 500, 0];    % Goal position
bs_loc = [900, 100, 20];      % Base station location
ris_loc = [200, 300, 40];     % RIS location

% Initialize channel metrics storage
channel_metrics = struct('time', [], 'direct_gain', [], 'cascaded_gain', []);

% Initialize OFDM transmission for a single time step
function [received_signal, direct_channel, cascaded_channel, Pb] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos, W)
    % Generate transmit data for each subcarrier
    
    Mb = 64;        % Number of OFDM subcarriers
    Nb = 8;         % Number of BS antennas
    Nr = 64;        % Number of RIS elements (8x8)
    Nt = 4;  
    
    x = zeros(Mb, 1);
    for n = 1:Mb
        % Generate complex Gaussian data with zero mean and unit variance
        x(n) = (randn + 1i*randn)/sqrt(2);
    end
    
    % Generate beamforming matrix if not provided
    if nargin < 4
        W = zeros(Nb, Mb);
        for i = 1:Mb
            w = (randn(Nb,1) + 1i*randn(Nb,1))/sqrt(2*Nb);
            W(:,i) = w/norm(w); % Normalize each beamforming vector
        end
    end
    
    % Verify transmit power constraint
    Pb = norm(W*x)^2;
    
    % Generate channels
    % Direct channel (BS to Vehicle): Nt x Nb
    direct_channel = generate_direct_channel(bs_loc, vehicle_pos, Nb, Nt);
    
    % Cascaded channel components
    % H_br (RIS to BS): Nr x Nb
    % H_rv (Vehicle to RIS): Nt x Nr
    [H_br, H_rv] = generate_cascaded_channel(bs_loc, ris_loc, vehicle_pos, Nb, Nr, Nt);
    
    % RIS phase shifts (diagonal matrix Nr x Nr)
    Phi = diag(exp(1i*2*pi*rand(Nr,1)));
    
    % Complete cascaded channel
    cascaded_channel = H_rv * Phi * H_br; % Results in Nt x Nb
    
    % Compute received signal for each subcarrier
    received_signal = zeros(Nt, Mb);
    for n = 1:Mb
        % Direct path (Nt x 1)
        y_direct = direct_channel * (W(:,n) * x(n));
        
        % RIS-assisted path (Nt x 1)
        y_ris = cascaded_channel * (W(:,n) * x(n));
        
        % Combine paths (Nt x 1)
        received_signal(:,n) = y_direct + y_ris;
    end
end

% Function to generate direct channel (Nt x Nb)
function H_d = generate_direct_channel(bs_loc, vehicle_pos, Nb, Nt)
    % Calculate distance
    d = norm(vehicle_pos - bs_loc);
    
    % Path loss (simplified model)
    PL = 1/sqrt(d^2);
    
    % Rayleigh fading
    H_d = PL * (randn(Nt,Nb) + 1i*randn(Nt,Nb))/sqrt(2);
end

% Function to generate cascaded channel components
function [H_br, H_rv] = generate_cascaded_channel(bs_loc, ris_loc, vehicle_pos, Nb, Nr, Nt)
    % Calculate distances
    d_br = norm(ris_loc - bs_loc);
    d_rv = norm(vehicle_pos - ris_loc);
    
    % Path loss
    PL_br = 1/sqrt(d_br^2);
    PL_rv = 1/sqrt(d_rv^2);
    
    % Generate channels with Rician fading (assuming stronger LoS component)
    K_factor = 10; % Rician K-factor
    
    % BS-RIS channel (Nr x Nb)
    H_los_br = generate_los_component(bs_loc, ris_loc, Nr, Nb);
    H_nlos_br = (randn(Nr,Nb) + 1i*randn(Nr,Nb))/sqrt(2);
    H_br = PL_br * sqrt(K_factor/(K_factor+1)) * H_los_br + ...
           PL_br * sqrt(1/(K_factor+1)) * H_nlos_br;
    
    % RIS-Vehicle channel (Nt x Nr)
    H_los_rv = generate_los_component(ris_loc, vehicle_pos, Nt, Nr);
    H_nlos_rv = (randn(Nt,Nr) + 1i*randn(Nt,Nr))/sqrt(2);
    H_rv = PL_rv * sqrt(K_factor/(K_factor+1)) * H_los_rv + ...
           PL_rv * sqrt(1/(K_factor+1)) * H_nlos_rv;
end

% Function to generate LoS component based on geometry
function H_los = generate_los_component(tx_loc, rx_loc, Nr, Nt)
    % Calculate angles
    diff_vec = rx_loc - tx_loc;
    [az, el] = cart2sph(diff_vec(1), diff_vec(2), diff_vec(3));
    
    % Generate steering vectors
    H_los = exp(1i*pi*[0:Nr-1]'*sin(az)) * exp(1i*pi*[0:Nt-1]*sin(el));
    H_los = H_los/sqrt(Nr*Nt); % Normalize
end

% Create figure
figure('Name', '3D Vehicle Simulator with Antenna Arrays');
hold on;
grid on;
axis([0 x_size 0 y_size 0 z_size]);
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Vehicle Motion Simulation with Antenna Arrays');

% Plot start and goal positions
plot3(vehicle_pos(1), vehicle_pos(2), vehicle_pos(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(goal_pos(1), goal_pos(2), goal_pos(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Initialize vehicle array plot
vehicle_array_markers = plot3(vehicle_array_positions(:,1), vehicle_array_positions(:,2), vehicle_array_positions(:,3), 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'b');

% Draw lines connecting centers
vehicle_to_bs = plot3([vehicle_pos(1), bs_loc(1)], [vehicle_pos(2), bs_loc(2)], [vehicle_pos(3), bs_loc(3)], '--k');
vehicle_to_ris = plot3([vehicle_pos(1), ris_loc(1)], [vehicle_pos(2), ris_loc(2)], [vehicle_pos(3), ris_loc(3)], '--m');

legend('BS Array', 'BS Center', 'RIS Array', 'RIS Center', 'Start', 'Goal', 'Vehicle Array');

% Set view angle for better 3D visualization
view(45, 30);

time = 0;
idx = 1;
while vehicle_pos(1) < goal_pos(1)
    % Update vehicle center position
    vehicle_pos(1) = vehicle_pos(1) + speed * dt;
    time = time + dt;
    
    % Update vehicle array positions
    for i = 1:Nt
        vehicle_array_positions(i,:) = vehicle_pos + [0, (i-1)*d, 0];
    end
    
    % Calculate OFDM transmission and channel characteristics
    [received_signal, direct_channel, cascaded_channel, ~] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos);
    
    % Store channel metrics
    channel_metrics.time(idx) = time;
    channel_metrics.direct_gain(idx) = 20*log10(norm(direct_channel));
    channel_metrics.cascaded_gain(idx) = 20*log10(norm(cascaded_channel));
    
    % Update vehicle array visualization
    set(vehicle_array_markers, 'XData', vehicle_array_positions(:,1), ...
                              'YData', vehicle_array_positions(:,2), ...
                              'ZData', vehicle_array_positions(:,3));
    
    % Update connection lines
    set(vehicle_to_bs, 'XData', [vehicle_pos(1), bs_loc(1)], ...
                      'YData', [vehicle_pos(2), bs_loc(2)], ...
                      'ZData', [vehicle_pos(3), bs_loc(3)]);
    set(vehicle_to_ris, 'XData', [vehicle_pos(1), ris_loc(1)], ...
                       'YData', [vehicle_pos(2), ris_loc(2)], ...
                       'ZData', [vehicle_pos(3), ris_loc(3)]);
    
    % Force MATLAB to draw the update
    drawnow;
    
    % Create real-time plot of channel gains (in a separate figure)
    if mod(idx, 10) == 0  % Update plot every 10 steps to reduce computation
        figure(2);
        plot(channel_metrics.time, channel_metrics.direct_gain, 'b-', ...
             channel_metrics.time, channel_metrics.cascaded_gain, 'r-');
        grid on;
        xlabel('Time (s)');
        ylabel('Channel Gain (dB)');
        legend('Direct Channel', 'Cascaded Channel');
        title('Channel Gains vs. Time');
    end
    
    % Add small pause to make motion visible
    pause(dt);
    
    idx = idx + 1;
end

% Final analysis plots
figure(3);
subplot(2,1,1);
plot(channel_metrics.time, channel_metrics.direct_gain, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Gain (dB)');
title('Direct Channel Gain vs. Time');

subplot(2,1,2);
plot(channel_metrics.time, channel_metrics.cascaded_gain, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Gain (dB)');
title('Cascaded Channel Gain vs. Time');

[received_signal, direct_channel, cascaded_channel, ~] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos);

% Calculate and display channel metrics
direct_channel_gain = 20*log10(norm(direct_channel));
cascaded_channel_gain = 20*log10(norm(cascaded_channel));

fprintf('Direct channel gain: %.2f dB\n', direct_channel_gain);
fprintf('Cascaded channel gain: %.2f dB\n', cascaded_channel_gain);

% Display matrix dimensions for verification
fprintf('\nMatrix dimensions:\n');
fprintf('Direct channel: %d x %d\n', size(direct_channel));
fprintf('Cascaded channel: %d x %d\n', size(cascaded_channel));
fprintf('Received signal: %d x %d\n', size(received_signal));