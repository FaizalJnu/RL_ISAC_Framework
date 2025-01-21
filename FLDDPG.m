classdef FLDDPG
    properties
        % Network parameters
        criticNet
        actorNet
        targetCriticNet
        targetActorNet
        
        % Memory buffer
        replayMemory
        maxMemorySize = 10000
        miniBatchSize = 64
        
        % Training parameters
        gamma = 0.99  % Discount factor
        tau = 0.001   % Soft update parameter
        epsilon = 1.0  % Initial exploration rate
        epsilonMin = 0.01
        epsilonDecay = 0.995
        
        % Learning rates
        actorLR = 0.0001
        criticLR = 0.001

        env
    end
    
    methods
        function obj = FLDDPG(stateDim, actionDim)
            % Initialize networks
            obj.criticNet = obj.createCriticNetwork(stateDim, actionDim);
            obj.actorNet = obj.createActorNetwork(stateDim, actionDim);
            obj.targetCriticNet = obj.createCriticNetwork(stateDim, actionDim);
            obj.targetActorNet = obj.createActorNetwork(stateDim, actionDim);
            
            % Initialize replay memory
            obj.replayMemory = [];
        end
        
        function net = createCriticNetwork(~, stateDim, actionDim)
            layers = [
                featureInputLayer(stateDim + actionDim, 'Name', 'input')
                fullyConnectedLayer(400, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(300, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(1, 'Name', 'output')
            ];
            net = dlnetwork(layerGraph(layers));
        end
        
        function net = createActorNetwork(~, stateDim, actionDim)
            layers = [
                featureInputLayer(stateDim, 'Name', 'input')
                fullyConnectedLayer(400, 'Name', 'fc1')
                reluLayer('Name', 'relu1')
                fullyConnectedLayer(300, 'Name', 'fc2')
                reluLayer('Name', 'relu2')
                fullyConnectedLayer(actionDim, 'Name', 'output')
                tanhLayer('Name', 'tanh')
            ];
            net = dlnetwork(layerGraph(layers));
        end
        
        function action = getAction(obj, state, noise)
            % Convert state to dlarray
            state = dlarray(state', 'CB');
            
            % Get action from actor network
            action = forward(obj.actorNet, state);
            action = extractdata(action)';
            
            % Add exploration noise
            if obj.epsilon > rand()
                action = action + obj.epsilon * noise;
                action = max(min(action, 1), -1);  % Clip action
            end
        end
        
        function obj = train(obj, episode, maxEpisodes, maxSteps)
            for e = 1:maxEpisodes
                % Reset environment and get initial state
                state = obj.resetEnvironment();
                
                for t = 1:maxSteps
                    % Update exploration rate
                    obj.epsilon = max(obj.epsilonMin, ...
                        obj.epsilon * obj.epsilonDecay);
                    
                    % Generate exploration noise
                    noise = randn(size(state));
                    
                    % Select action
                    action = obj.getAction(state, noise);
                    
                    % Execute action and observe reward and next state
                    [nextState, reward] = obj.executeAction(action);
                    
                    % Store transition in replay memory
                    obj.storeTransition(state, action, reward, nextState);
                    
                    % Train if enough samples are available
                    if size(obj.replayMemory, 1) >= obj.miniBatchSize
                        obj = obj.updateNetworks();
                    end
                    
                    state = nextState;
                end
            end
        end
        
        function obj = updateNetworks(obj)
            % Sample mini-batch
            idx = randperm(size(obj.replayMemory, 1), obj.miniBatchSize);
            batch = obj.replayMemory(idx, :);
            
            states = dlarray(batch(:, 1:end-3)', 'CB');
            actions = dlarray(batch(:, end-2)', 'CB');
            rewards = dlarray(batch(:, end-1)', 'CB');
            nextStates = dlarray(batch(:, end)', 'CB');
            
            % Update critic
            targetActions = forward(obj.targetActorNet, nextStates);
            targetQ = forward(obj.targetCriticNet, [nextStates; targetActions]);
            yj = rewards + obj.gamma * targetQ;
            
            [gradients, loss] = dlfeval(@criticLoss, obj.criticNet, ...
                states, actions, yj);
            obj.criticNet = adamupdate(obj.criticNet, gradients, ...
                [], [], [], obj.criticLR);
            
            % Update actor
            [gradients, ~] = dlfeval(@actorLoss, obj.actorNet, ...
                obj.criticNet, states);
            obj.actorNet = adamupdate(obj.actorNet, gradients, ...
                [], [], [], obj.actorLR);
            
            % Soft update target networks
            obj = obj.updateTargetNetworks();
        end
        
        function obj = updateTargetNetworks(obj)
            % Soft update target networks
            criticWeights = obj.criticNet.Learnables;
            targetCriticWeights = obj.targetCriticNet.Learnables;
            
            for i = 1:numel(criticWeights)
                targetCriticWeights{i}.Value = ...
                    (1 - obj.tau) * targetCriticWeights{i}.Value + ...
                    obj.tau * criticWeights{i}.Value;
            end
            
            actorWeights = obj.actorNet.Learnables;
            targetActorWeights = obj.targetActorNet.Learnables;
            
            for i = 1:numel(actorWeights)
                targetActorWeights{i}.Value = ...
                    (1 - obj.tau) * targetActorWeights{i}.Value + ...
                    obj.tau * actorWeights{i}.Value;
            end
        end
        
        function storeTransition(obj, state, action, reward, nextState)
            transition = [state, action, reward, nextState];
            obj.replayMemory = [obj.replayMemory; transition];
            
            if size(obj.replayMemory, 1) > obj.maxMemorySize
                obj.replayMemory(1, :) = [];
            end
        end
    end
end

% Loss functions (defined as local functions)
function [gradients, loss] = criticLoss(criticNet, states, actions, targets)
    Q = forward(criticNet, [states; actions]);
    loss = mean((Q - targets).^2);
    gradients = dlgradient(loss, criticNet.Learnables);
end

function [gradients, loss] = actorLoss(actorNet, criticNet, states)
    actions = forward(actorNet, states);
    Q = forward(criticNet, [states; actions]);
    loss = -mean(Q);
    gradients = dlgradient(loss, actorNet.Learnables);
end