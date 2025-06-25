# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Initialize a score with the current successor game state's score.
        score = successorGameState.getScore()

        # Get the list of food pellets in the successor state.
        foodList = newFood.asList()

        # --- Food Considerations ---
        # If there's food, calculate the distance to the closest food.
        # The closer the food, the higher the bonus.
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            # Add the inverse of the distance as a bonus. A smaller distance yields a larger bonus.
            score += 1.0 / (minFoodDistance + 1)  # Add 1 to avoid division by zero.

        # Add a penalty for remaining food. Fewer food pellets mean a higher score.
        # This incentivizes eating food.
        score -= len(foodList) * 10

        # --- Ghost Considerations ---
        # Iterate through each ghost to evaluate its impact.
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            # Calculate Manhattan distance to the ghost.
            ghostDistance = manhattanDistance(newPos, ghostPos)
            scaredTime = newScaredTimes[i]

            if scaredTime > 0:
                # If the ghost is scared, Pacman should try to eat it.
                # A smaller distance to a scared ghost gives a large bonus.
                # The bonus is higher if the scared time is also high, allowing for more time to eat.
                if ghostDistance <= scaredTime:  # Only consider eating if within scared time
                    score += 200  # Large bonus for eating a scared ghost.
                score -= ghostDistance  # Still prefer closer scared ghosts
            else:
                # If the ghost is not scared, Pacman should avoid it.
                # A smaller distance to a non-scared ghost results in a large penalty.
                if ghostDistance <= 1:  # If very close to a non-scared ghost (e.g., adjacent)
                    score -= 500  # Severe penalty for being too close to an active ghost.
                elif ghostDistance <= 3:  # If somewhat close
                    score -= 50  # Moderate penalty

        # --- Capsule Considerations ---
        # Evaluate power pellets (capsules).
        # Eating a capsule provides a score bonus. The inverse of the distance
        # to the closest capsule can also be a factor to encourage moving towards them.
        newCapsules = successorGameState.getCapsules()
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            # If Pacman eats food, provide a small bonus
            score += 10
        if len(currentGameState.getCapsules()) > len(newCapsules):
            # If Pacman eats a capsule, provide a significant bonus
            score += 1000  # Large bonus for eating a capsule

        # Prefer to eat all capsules.
        score -= len(newCapsules) * 50

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # This is the entry point for the Minimax Agent.
        # It calls the minimax function for Pacman (agentIndex = 0) at depth 0.
        # The result of minimax will be the best action for Pacman.
        action, _ = self.minimax(gameState, 0, 0)
        return action

    def minimax(self, gameState: GameState, agentIndex: int, currentDepth: int):
        """
        The core minimax algorithm.
        :param gameState: The current game state.
        :param agentIndex: The index of the current agent (0 for Pacman, 1+ for ghosts).
        :param currentDepth: The current depth in the search tree.
        :return: A tuple of (best_action, best_score) for the current agent.
        """

        # If we have reached the maximum depth or a terminal state (win/lose),
        # evaluate the state using the provided evaluation function.
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # Get the total number of agents in the game.
        numAgents = gameState.getNumAgents()

        # Calculate the next agent's index and the next depth.
        # If the current agent is the last ghost, the next agent is Pacman, and depth increases.
        # Otherwise, it's the next ghost in the sequence.
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth

        # Get all legal actions for the current agent.
        legalActions = gameState.getLegalActions(agentIndex)

        # If there are no legal actions, evaluate the current state as a terminal state.
        if not legalActions:
            return None, self.evaluationFunction(gameState)

        # --- Pacman's Turn (Maximizer) ---
        if agentIndex == 0:  # Pacman is the maximizing agent
            bestScore = float('-inf')
            bestAction = None
            for action in legalActions:
                # Generate the successor state after Pacman takes an action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call minimax for the next agent.
                _, score = self.minimax(successorState, nextAgentIndex, nextDepth)
                # If this action leads to a better score, update bestScore and bestAction.
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            return bestAction, bestScore

        # --- Ghost's Turn (Minimizer) ---
        else:  # Ghosts are minimizing agents
            bestScore = float('inf')
            bestAction = None
            for action in legalActions:
                # Generate the successor state after a ghost takes an action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call minimax for the next agent.
                _, score = self.minimax(successorState, nextAgentIndex, nextDepth)
                # If this action leads to a worse score, update bestScore and bestAction.
                if score < bestScore:
                    bestScore = score
                    bestAction = action
            return bestAction, bestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initialize alpha and beta for the root node (Pacman's turn).
        # Alpha is the best score found so far for the maximizing player (Pacman).
        # Beta is the best score found so far for the minimizing players (Ghosts).
        alpha = float('-inf')
        beta = float('inf')

        # Start the alpha-beta pruning search from Pacman (agentIndex 0) at depth 0.
        action, _ = self.alphaBeta(gameState, 0, 0, alpha, beta)
        return action

    def alphaBeta(self, gameState: GameState, agentIndex: int, currentDepth: int, alpha: float, beta: float):
        """
        The core alpha-beta pruning algorithm.
        :param gameState: The current game state.
        :param agentIndex: The index of the current agent (0 for Pacman, 1+ for ghosts).
        :param currentDepth: The current depth in the search tree.
        :param alpha: The alpha value (best score found so far for max player).
        :param beta: The beta value (best score found so far for min player).
        :return: A tuple of (best_action, best_score) for the current agent.
        """

        # If we have reached the maximum depth or a terminal state (win/lose),
        # evaluate the state using the provided evaluation function.
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # Get the total number of agents in the game.
        numAgents = gameState.getNumAgents()

        # Calculate the next agent's index and the next depth.
        # If the current agent is the last ghost, the next agent is Pacman, and depth increases.
        # Otherwise, it's the next ghost in the sequence.
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth

        # Get all legal actions for the current agent.
        legalActions = gameState.getLegalActions(agentIndex)

        # If there are no legal actions, evaluate the current state as a terminal state.
        if not legalActions:
            return None, self.evaluationFunction(gameState)

        # --- Pacman's Turn (Maximizer) ---
        if agentIndex == 0:  # Pacman is the maximizing agent
            bestScore = float('-inf')
            bestAction = None
            for action in legalActions:
                # Generate the successor state after Pacman takes an action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call alphaBeta for the next agent.
                _, score = self.alphaBeta(successorState, nextAgentIndex, nextDepth, alpha, beta)
                # Update bestScore and bestAction.
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                # Pruning condition: if bestScore is greater than or equal to beta,
                # then this path will not be chosen by the minimizing player above,
                # so we can stop exploring further.
                if bestScore >= beta:
                    return bestAction, bestScore
                # Update alpha: the best score found so far for the maximizing player.
                alpha = max(alpha, bestScore)
            return bestAction, bestScore

        # --- Ghost's Turn (Minimizer) ---
        else:  # Ghosts are minimizing agents
            bestScore = float('inf')
            bestAction = None
            for action in legalActions:
                # Generate the successor state after a ghost takes an action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call alphaBeta for the next agent.
                _, score = self.alphaBeta(successorState, nextAgentIndex, nextDepth, alpha, beta)
                # Update bestScore and bestAction.
                if score < bestScore:
                    bestScore = score
                    bestAction = action
                # Pruning condition: if bestScore is less than or equal to alpha,
                # then this path will not be chosen by the maximizing player above,
                # so we can stop exploring further.
                if bestScore <= alpha:
                    return bestAction, bestScore
                # Update beta: the best score found so far for the minimizing player.
                beta = min(beta, bestScore)
            return bestAction, bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # This is the entry point for the Expectimax Agent.
        # It calls the expectimax function for Pacman (agentIndex = 0) at depth 0.
        # The result of expectimax will be the best action for Pacman.
        action, _ = self.expectimax(gameState, 0, 0)
        return action

    def expectimax(self, gameState: GameState, agentIndex: int, currentDepth: int):
        """
        The core expectimax algorithm.
        :param gameState: The current game state.
        :param agentIndex: The index of the current agent (0 for Pacman, 1+ for ghosts).
        :param currentDepth: The current depth in the search tree.
        :return: A tuple of (best_action, best_score) for the current agent.
        """

        # If we have reached the maximum depth or a terminal state (win/lose),
        # evaluate the state using the provided evaluation function.
        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        # Get the total number of agents in the game.
        numAgents = gameState.getNumAgents()

        # Calculate the next agent's index and the next depth.
        # If the current agent is the last ghost, the next agent is Pacman, and depth increases.
        # Otherwise, it's the next ghost in the sequence.
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = currentDepth + 1 if nextAgentIndex == 0 else currentDepth

        # Get all legal actions for the current agent.
        legalActions = gameState.getLegalActions(agentIndex)

        # If there are no legal actions, evaluate the current state as a terminal state.
        if not legalActions:
            return None, self.evaluationFunction(gameState)

        # --- Pacman's Turn (Maximizer) ---
        if agentIndex == 0:  # Pacman is the maximizing agent
            bestScore = float('-inf')
            bestAction = None
            for action in legalActions:
                # Generate the successor state after Pacman takes an action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call expectimax for the next agent.
                _, score = self.expectimax(successorState, nextAgentIndex, nextDepth)
                # If this action leads to a better score, update bestScore and bestAction.
                if score > bestScore:
                    bestScore = score
                    bestAction = action
            return bestAction, bestScore

        # --- Ghost's Turn (Expectation) ---
        else:  # Ghosts are expectation agents (choosing uniformly at random)
            totalScore = 0
            for action in legalActions:
                # Generate the successor state after a ghost takes an action.
                successorState = gameState.generateSuccessor(agentIndex, action)
                # Recursively call expectimax for the next agent.
                _, score = self.expectimax(successorState, nextAgentIndex, nextDepth)
                # Add the score to the total.
                totalScore += score
            # The expected score is the average of scores from all legal actions.
            expectedScore = totalScore / len(legalActions)
            return None, expectedScore  # For expectation nodes, the action doesn't matter for the return value of the recursive call


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    This evaluation function aims to provide a more comprehensive assessment of the game state
    for Pacman by considering multiple factors:

    1.  **Current Score:** The base score of the game state is always included.
    2.  **Food Remaining:**
        * A penalty is applied for each piece of food remaining, incentivizing Pacman to eat food.
        * A bonus is given based on the inverse of the distance to the closest food. This encourages Pacman to move towards food.
    3.  **Capsules (Power Pellets):**
        * A significant bonus is given if a capsule is eaten, as it allows Pacman to eat ghosts.
        * A penalty is applied for remaining capsules to encourage eating them.
        * A bonus is given based on the inverse of the distance to the closest capsule.
    4.  **Ghosts:**
        * **Scared Ghosts:** If a ghost is scared, Pacman is heavily incentivized to move towards it to eat it. The closer the scared ghost, the higher the bonus. A larger scared time also gives a higher bonus.
        * **Active (Non-Scared) Ghosts:** Pacman is heavily penalized for being close to active ghosts. The closer the ghost, the larger the penalty. This encourages avoidance.
    5.  **Game State:** Checks for win/lose states and assigns very high/low scores accordingly to prioritize winning and avoid losing.

    The combination of these factors provides a robust evaluation that guides Pacman towards eating food and capsules while effectively managing ghost threats.
    """
    "*** YOUR CODE HERE ***"
    # Current score is a baseline.
    score = currentGameState.getScore()

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()

    # --- Food Considerations ---
    # Penalty for remaining food
    score -= len(foodList) * 20

    # Distance to closest food
    if foodList:
        minFoodDistance = min([manhattanDistance(pacmanPos, food) for food in foodList])
        score += 1.0 / (minFoodDistance + 1) * 10  # More emphasis on closer food

    # --- Capsule Considerations ---
    # Penalty for remaining capsules
    score -= len(capsules) * 100

    # Distance to closest capsule
    if capsules:
        minCapsuleDistance = min([manhattanDistance(pacmanPos, capsule) for capsule in capsules])
        score += 1.0 / (minCapsuleDistance + 1) * 50  # Incentivize getting capsules

    # --- Ghost Considerations ---
    for i, ghostState in enumerate(ghostStates):
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)
        scaredTime = scaredTimes[i]

        if scaredTime > 0:
            # If ghost is scared, highly incentivize eating it.
            # Bonus is higher if ghost is closer and has more scared time.
            if ghostDistance < scaredTime:  # Only consider if we can reach it while scared
                score += (scaredTime - ghostDistance) * 2  # Encourage eating scared ghosts
            else:  # If we can't reach it while scared, it's still good to move away from it slightly
                score += 1.0 / (
                            ghostDistance + 1) * 5  # Small bonus for being near, but not too near, a scard ghost that's far
        else:
            # If ghost is not scared, heavily penalize being close to it.
            if ghostDistance <= 1:  # Immediate danger
                score -= 1000
            elif ghostDistance <= 3:  # Close but not immediate
                score -= 100
            elif ghostDistance > 3:  # Far away
                score += 1.0 / (ghostDistance + 1) * 5  # Small bonus for being far from dangerous ghosts.

    # Win/Lose states
    if currentGameState.isWin():
        score += 10000
    if currentGameState.isLose():
        score -= 10000

    return score

# Abbreviation
better = betterEvaluationFunction
