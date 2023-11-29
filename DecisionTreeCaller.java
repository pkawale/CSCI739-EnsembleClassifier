import java.io.*;

public class DecisionTreeCaller {

    public static void main(String[] args) {
        try {
            // Command to run the Python script
            StringBuilder command = new StringBuilder("python3 randomForestClassifier.py");
            for (String arg : args) {
                command.append(" ").append(arg);
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