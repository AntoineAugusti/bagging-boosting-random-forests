% function classifier = decisionStumpTrain(X, Y, weights);
%   Input:
%   - X: N*M data matrix where N is the number of examples and M the number of features
%   - Y: N*1 label matrix (label -1 or 1)
%   - weights:  N*1 weight matrix (weights in [0..1])
%   Output:
%    - classifier: struct later used in testing
function classifier = decisionStumpTrain(X, Y, weights)
    nX = size(X, 1);
    nC = size(X, 2);

    idPos = find(Y == 1);
    idNeg = find(Y == -1);

    YPos = zeros(nX, 1);
    YPos(idPos) = weights(idPos);

    YNeg = zeros(nX, 1);
    YNeg(idNeg) = weights(idNeg);

    weightsPos = weights(idPos);
    weightsNeg = weights(idNeg);

    for c=1:nC
        data = X(:, c);

        mPos(c) = sum(data(idPos) .* weightsPos) / sum(weightsPos);
        mNeg(c) = sum(data(idNeg) .* weightsNeg) / sum(weightsNeg);
        sens(c) = sign(mPos(c) - mNeg(c));
        if sens(c) > 0
            [sdata, idx] = sort(data);
        else
            [sdata, idx] = sort(data, 1, 'descend');
        end
        sYPos = YPos(idx);
        sYNeg = YNeg(idx);
        errYPos = cumsum(sYPos);
        errYNeg = flipud(cumsum(flipud(sYNeg)));
        errTot = errYPos + errYNeg;
        [err(c), idx] = min(errTot);
        seuil(c) = sdata(idx);
    end

    [dum, c] = min(err);

    classifier.icarac = c;
    classifier.seuil = seuil(c);

    if sens(c) > 0
        classifier.sens = '<';
    else
        classifier.sens = '>';
    end
end
