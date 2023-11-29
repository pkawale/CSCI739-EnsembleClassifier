import java.io.*;

public class DecisionTreeCaller {

    public static void main(String[] args) {
        try {
            // Command to run the Python script
            StringBuilder command = new StringBuilder("python3 randomForestClassifier.py");

            for (int i = 0; i < args.length; i++) {
                String arg = args[i];

                if (arg.startsWith("-")) {
                    if (i + 1 < args.length) {
                        String value = args[i + 1];

                        // Validate the argument value
                        if (arg.equals("-n") || arg.equals("-s") || arg.equals("-d") || arg.equals("-f")) {
                            int intValue = Integer.parseInt(value);
                            if (intValue < 0) {
                                System.out.println("Invalid value for " + arg + ": " + intValue);
                                return;
                            }
                        } else if (arg.equals("-i")) {
                            if (!value.equals("gini") && !value.equals("entropy")) {
                                System.out.println("Invalid value for -i: " + value);
                                return;
                            }
                        }
                        command.append(" ").append(arg).append(" ").append(value);
                        i++;
                    } else {
                        System.out.println("Expected value after " + arg);
                        return;
                    }
                }
            }
            ProcessBuilder pb = new ProcessBuilder(command.toString().split(" "));
            Process p = pb.start();

            // Get the output from the script
            BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }

            // Wait for the process to finish and check the exit value
            int exitValue = p.waitFor();
            System.out.println("Process exited with code " + exitValue);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}