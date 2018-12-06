package distributed_perceptron;


import peersim.config.Configuration;
import peersim.core.Control;


public class observer implements Control {
	/** 
	 * String name of the parameter used to select the protocol to operate on
	 */
	public static final String PAR_PROTID = "protocol";
	/** The name of this object in the configuration file */
	private final String name;
	/** Protocol identifier */
	private final int pid;
	
	public boolean execute()
	{return false;}
   
	/**
	 * Creates a new observer and initializes the configuration parameter.
	 */
	public observer(String name) {
		this.name = name;
		pid = Configuration.getPid(name + "." + PAR_PROTID);
	  }


}
