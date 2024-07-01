from abc import ABC, abstractmethod
import numpy as np
import gurobipy as gp


class MultiAgentInteractionInterface(ABC):

    @abstractmethod
    def function(self, x):
        return x

    @abstractmethod
    def add_constraints(self, m, store_vars, arm):
        pass

class LogMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M, alpha=0.05):
        self.arm_id    :int = id
        self.log_base  :int = 2 + (id%10)
        self.alpha     :int = alpha
        self.M         :int = M

    def function(self, x):
        return (np.emath.logn(self.log_base, self.alpha*x + 1/(self.log_base)) + 1) / (np.emath.logn(self.log_base, self.alpha + 1/(self.log_base)) + 1)
    
    def add_constraints(self, m, store_vars):
        # This is the number of agents selecting each arm (call it x)
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, 
                                                     lb = 0.0, 
                                                     ub = self.M, 
                                                     name = f"x_{self.arm_id}"
                                                     )
        
        # This is alpha*x_{} + 1/(log_base)
        temp1                                = m.addVar(vtype = gp.GRB.CONTINUOUS, 
                                                        lb = 1/self.log_base, 
                                                        ub = self.M*self.alpha + 1/self.log_base, 
                                                        name = f"self.alpha*x_{self.arm_id}+1/(log_base)"
                                                        )
        
        # This is np.emath.logn(log_base, alpha*x + 1/(log_base))
        temp2                                = m.addVar(vtype = gp.GRB.CONTINUOUS, 
                                                        lb = -1, 
                                                        ub = np.emath.logn(self.log_base, self.alpha*self.M + 1/(self.log_base)),
                                                        name = f"(np.emath.logn(log_base,self.alpha*x_{self.arm_id}+1/(log_base)))"
                                                        )
        
        # This is f(x) = (np.emath.logn(log_base, alpha*x + 1/(log_base)) + 1) / (np.emath.logn(log_base, alpha + 1/(log_base)) + 1)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, 
                                                     lb = 0, 
                                                     ub = (np.emath.logn(self.log_base, self.alpha*self.M + 1/(self.log_base)) + 1) / (np.emath.logn(self.log_base, self.alpha + 1/(self.log_base)) + 1),
                                                     name = f"f(x_{self.arm_id})"
                                                     )


        # Add constraints
        m.addConstr(temp1 == self.alpha * store_vars[f"x_{self.arm_id}"] + 1/self.log_base, name = f"constr1_x_{self.arm_id}")
        m.addGenConstrLogA(temp1, temp2, self.log_base)
        m.addConstr(store_vars[f"f(x_{self.arm_id})"] == (temp2+1)/(np.emath.logn(self.log_base, self.alpha + 1/self.log_base) + 1), name = f"constr2_x_{self.arm_id}")

        return store_vars

class LinearMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M):
        self.arm_id    :int = id
        self.M         :int = M

    def function(self, x):
        return x
    
    def add_constraints(self, m, store_vars):
        # Number of agents at arm arm_id
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # Transformation function
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")

        # Constrain transformation == number of agents
        m.addConstr(store_vars[f"f(x_{self.arm_id})"] == store_vars[f"x_{self.arm_id}"], name = f"constr_x_{self.arm_id}")

        return store_vars
    
class ConstantMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M):
        self.arm_id    :int = id
        self.M         :int = M

    def function(self, x):
        if x > 0 :
            return 1
        else:
            return x
    
    def add_constraints(self, m, store_vars):

        # Number of agents at specific arm--denote by x
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # Transformation f(c)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")

        # Indicator of x > 0
        b = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{self.arm_id}>0")
       
        # Add constraints
        eps = 0.0001
        M = self.M + eps
        # If b = 0, this is non-binding
        # If b = 1, this means x >= 0.001 -> x >= 1 because x is an integer
        m.addConstr(store_vars[f"x_{self.arm_id}"] >= eps - M * (1 - b), name=f"constr_x_{self.arm_id}_b=1")
        # If b = 0, x <= 0, so x = 0
        # If b = 1, this is non-binding
        m.addConstr(store_vars[f"x_{self.arm_id}"] <= M * b, name=f"constr_x_{self.arm_id}_b=0")

        # If b == 1, then constrain f(x) == 1
        # If b == 0, then constrain f(x) == x == 0
        m.addConstr((b == 1) >> (store_vars[f"f(x_{self.arm_id})"] == 1), name = f"constr_f(x_{self.arm_id})_b=1")
        m.addConstr((b == 0) >> (store_vars[f"f(x_{self.arm_id})"] == store_vars[f"x_{self.arm_id}"]), name = f"constr_f(x_{self.arm_id})_b=0")

        return store_vars



class MuchMoreLogMultiAgentInteraction(LogMultiAgentInteraction):
    def __init__(self, id, M, log_base_shift = 98):
        # Call concave, but shift log base by log_base_shift
        super().__init__(id, M)
        # Re-define log base to log base + shift
        self.log_base += log_base_shift

class CollisionMultiAgentInteraction(MultiAgentInteractionInterface):
    def __init__(self, id, M):
        self.arm_id    :int = id
        self.M         :int = M

    def function(self, x):
        return x if x <= 1 else 0
    
    def add_constraints(self, m, store_vars):
        # This is the number of agents selecting each arm (call it x)
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # This is f(c)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")
        # Indicator x > 1
        b = m.addVar(vtype=gp.GRB.BINARY, name=f"x_{self.arm_id}>1")
        # Constants

        # M is chosen to be as small as possible given the bounds on x and y
        eps = 0.0001
        M = self.M + eps
        # If b = 1, x > 1 (so x >= 2)
        # If b = 0, this is non-binding
        m.addConstr(store_vars[f"x_{self.arm_id}"] >= 1 + eps - M * (1 - b), name=f"constr_x_{self.arm_id}_b=1")
        # If b = 0, then x <= 1
        # If b = 1, then this is non-binding
        m.addConstr(store_vars[f"x_{self.arm_id}"] <= 1 + M * b, name=f"constr_x_{self.arm_id}_b=0")
        
        # Add constraints
        m.addConstr((b == 1) >> (store_vars[f"f(x_{self.arm_id})"] == 0), name = f"constr_f(x_{self.arm_id})_b=1")
        m.addConstr((b == 0) >> (store_vars[f"f(x_{self.arm_id})"] == store_vars[f"x_{self.arm_id}"]), name = f"constr_f(x_{self.arm_id})_b=0")
        return store_vars


class PowerMultiAgentInteraction(MultiAgentInteractionInterface):
    """
        A MultiAgentInteractionInterface that implements the transformation function with numerator and denominator as specified by the user
    """
    def __init__(self, id, M, numer = 1, denom = 2):
        try:
            assert(numer >= 0 and denom >= 0)
        except:
            raise ValueError("Numerator must be non-negative and less than the denominator")
        
        self.arm_id    :int = id
        self.M         :int = M
        self.numer     :int = numer
        self.denom     :int = denom

    def function(self, x):
        return x**(self.numer / self.denom)
    
    def add_constraints(self, m, store_vars):
        # This is the number of agents selecting each arm (call it x)
        store_vars[f"x_{self.arm_id}"]    = m.addVar(vtype = gp.GRB.INTEGER, lb = 0.0, ub = self.M, name = f"x_{self.arm_id}")
        # This is f(c)
        store_vars[f"f(x_{self.arm_id})"] = m.addVar(vtype = gp.GRB.CONTINUOUS, name = f"f(x_{self.arm_id})")

        # Add constraint f(c) = c^(numer/denom)
        m.addGenConstrPow(store_vars[f"x_{self.arm_id}"], store_vars[f"f(x_{self.arm_id})"], self.numer / self.denom, name = f"constr_x_{self.arm_id}")

        return store_vars

def getFunction(id, function_type, params):
    if function_type == 'log':
        return LogMultiAgentInteraction(id, params.M)
    elif function_type == 'collision':
        return CollisionMultiAgentInteraction(id, params.M)
    elif function_type == 'more_log':
        return MuchMoreLogMultiAgentInteraction(id, params.M)
    elif function_type == 'linear':
        return LinearMultiAgentInteraction(id, params.M)
    elif function_type == 'constant':
        return ConstantMultiAgentInteraction(id, params.M)
    elif function_type.startswith('power'):
        numer, denom = [int(x) for x in function_type.split('_')[1:]]
        return PowerMultiAgentInteraction(id, params.M, numer, denom)
    else:
        return None