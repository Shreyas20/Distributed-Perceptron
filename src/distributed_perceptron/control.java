package distributed_perceptron;

//import javax.naming.ldap.Control;

import peersim.config.Configuration;
import peersim.core.Network;
import peersim.core.Control;

//import peersim.perceptronNode;

public class control implements Control{
	/** 
	 * String name of the parameter used to select the protocol to operate on
	 */
	public static final String PAR_PROTID = "protocol";

	/** The name of this object in the configuration file */
	private final String name;

	/** Protocol identifier */
	private final int pid;

	// iterator counter
	private static int i = 0;
	/**
	 * Creates a new observer and initializes the configuration parameter.
	 */
	public control(String name) {
		this.name = name;
		this.pid = Configuration.getPid(name + "." + PAR_PROTID);
	  }

	//--------------------------------------------------------------------------
	// Methods
	//--------------------------------------------------------------------------

	// Comment inherited from interface
	// Do nothing, just for test
	public boolean execute() {
		final int len = Network.size();
		for (int i = 0; i <  len; i++) {
			perceptronNode node = (perceptronNode) Network.get(i);
			//node.writeLossNorm();
		}
		System.out.println("Running final control");
		return false;
	}
	

}
