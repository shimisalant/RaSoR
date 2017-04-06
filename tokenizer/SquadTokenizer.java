
import static java.lang.Math.toIntExact;

import java.io.IOException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.StringReader;

import java.util.Arrays;
import java.util.List;
import java.util.LinkedList;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;
import java.util.Iterator;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

// http://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/json-simple/json-simple-1.1.1.jar
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

// http://argparse4j.github.io/
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;

// http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/process/PTBTokenizer.html
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;


public class SquadTokenizer {

  private static Charset CHARSET = StandardCharsets.UTF_8;

  private static String TOK_OPTS = "invertible=true,untokenizable=noneKeep,normalizeParentheses=false,normalizeOtherBrackets=false";

  private static class TokenizedText {
    List<String> tokens = new LinkedList<String>();
    List<String> originals = new LinkedList<String>();
    List<String> whitespaceAfters = new LinkedList<String>();
    List<Integer> startCharIdxs = new LinkedList<Integer>();
    List<Integer> afterEndCharIdxs = new LinkedList<Integer>();
  }

  private static class JsonMetadata {
    Set<String> allTokens = new HashSet<String>();
    Set<String> unknownTokens = new HashSet<String>();
    int numArticles = 0;
    int numInvalidArticles = 0;
    int numParagraphs = 0;
    int numInvalidParagraphs = 0;
    int numQuestions = 0;
    int numFullQuestions = 0;
    int numPartialQuestions = 0;
    int numInvalidQuestions = 0;
    int numAnswers = 0;
    int numInvalidAnswers = 0;
    int numSplits = 0;
    @Override
    public String toString() {
      return "word-types: " + allTokens.size() +
        "\nunknown word-types: " + unknownTokens.size() +
        "\n\t(invalidity is due to failure to match answer string to word tokens):" +
        "\narticles: " + numArticles + " (invalid: " + numInvalidArticles + ")" +
        "\nparagraphs: " + numParagraphs + " (invalid: " + numInvalidParagraphs + ")" +
        "\nquestions: " + numQuestions + " (invalid: " + numInvalidQuestions +
        ", some answers: " + numPartialQuestions + ", all answers: " + numFullQuestions + ")" +
        "\nanswers: " + numAnswers + " (invalid: " + numInvalidAnswers + ")" +
        "\nnum of performed splits: " + numSplits;
    }
  }


  public static void main(String[] args) throws IOException, ParseException {
    ArgumentParser parser = ArgumentParsers.newArgumentParser("SquadTokenizer").defaultHelp(true);
    parser.addArgument("in_json").help("input JSON file");
    parser.addArgument("out_json").help("output JSON file");
    parser.addArgument("--words_txt").required(true).help(
      "text file listing all known words (from GloVe)");
    parser.addArgument("--has_answers").action(Arguments.storeTrue()).help(
      "whether input JSON contains answers (as in train / dev set)");
    parser.addArgument("--split").action(Arguments.storeTrue()).help(
      "whether to split hyphenated words, when constituent tokens are found in GloVe");
    parser.addArgument("--verbose").action(Arguments.storeTrue());

    Namespace ns = parser.parseArgsOrFail(args);
    // System.out.println(ns);
    String inJson = ns.getString("in_json");
    String outJson = ns.getString("out_json");
    boolean hasAnswers = ns.getBoolean("has_answers");
    String wordsTxt = ns.getString("words_txt");
    boolean split = ns.getBoolean("split");
    boolean verbose = ns.getBoolean("verbose");

    System.out.println("Reading known words from " + wordsTxt);
    Set<String> knownWords = readKnownWords(wordsTxt);
    System.out.println("Writing tokenized version of " + inJson + " to " + outJson);
    JsonMetadata jsonMetadata = writeTokenizedJson(inJson, outJson, knownWords, hasAnswers, split, verbose);
    System.out.println(jsonMetadata + "\n");
  }


  private static Set<String> readKnownWords(String inFilename) throws IOException {
    Set<String> knownWords= new HashSet<String>();
    try (
      BufferedReader fin = new BufferedReader(new InputStreamReader(new FileInputStream(inFilename), CHARSET));
    ) {
      for(String line; (line = fin.readLine()) != null; ) {
        knownWords.add(line); // readLine removes EOL chars
      }
    }
    System.out.println("Number of known words: " + knownWords.size() + "\n");
    return knownWords;
  }


  private static JsonMetadata writeTokenizedJson(
    String inFilename, String outFilename, Set<String> knownWords, boolean hasAnswers, boolean split, boolean verbose)
      throws IOException, ParseException {
    try (
      BufferedReader fin = new BufferedReader(new InputStreamReader(new FileInputStream(inFilename), CHARSET));
      BufferedWriter fout = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFilename), CHARSET))
    ) {
      JSONObject json = (JSONObject) new JSONParser().parse(fin);
      JsonMetadata jsonMetadata = addTokenData(json, knownWords, hasAnswers, split, verbose);
      fout.write(json.toJSONString());
      return jsonMetadata;
    }
  }


  @SuppressWarnings("unchecked")
  private static JsonMetadata addTokenData(
    JSONObject json, Set<String> knownWords, boolean hasAnswers, boolean split, boolean verbose) {
    JsonMetadata jsonMetadata = new JsonMetadata();

    Set<String> unknownWords = new HashSet<String>();
    // String version = (String) json.get("version");
    JSONArray articleObjs = (JSONArray) json.get("data");
    if (articleObjs.isEmpty()) {
      throw new RuntimeException("No articles");
    }

    int numArticles = articleObjs.size();
    int numInvalidArticles = 0;
    for (Object articleObj : articleObjs) {
      JSONObject articleJson = (JSONObject) articleObj;
      String title = (String) articleJson.get("title");
      JSONArray paragraphObjs = (JSONArray) articleJson.get("paragraphs");
      if (paragraphObjs.isEmpty()) {
        throw new RuntimeException("No paragraphs\narticle:\n" + title);
      }

      int numParagraphs = paragraphObjs.size();
      int numInvalidParagraphs = 0;
      for (Object paragraphObj : paragraphObjs) {
        JSONObject paragraphJson = (JSONObject) paragraphObj;

        String contextStr = (String) paragraphJson.get("context");
        TokenizedText contextTok = tokenize(contextStr, knownWords, unknownWords, split, jsonMetadata, verbose);
        assertReconstruction("context", contextStr, contextTok, -1, -1);
        putTokens(paragraphJson, contextTok);
        jsonMetadata.allTokens.addAll(contextTok.tokens);
        JSONArray qaObjs = (JSONArray) paragraphJson.get("qas");
        if (qaObjs.isEmpty()) {
          throw new RuntimeException("No questions\narticle:\n" + title + "\ncontext:\n" + contextStr);
        }

        int numQuestions = qaObjs.size();
        int numFullQuestions = 0;
        int numPartialQuestions = 0;
        int numInvalidQuestions = 0;
        for (Object qaObj : qaObjs) {
          JSONObject qaJson = (JSONObject) qaObj;
          String idStr = (String) qaJson.get("id");
          String questionStr = (String) qaJson.get("question");
          TokenizedText questionTok = tokenize(questionStr, knownWords, unknownWords, split, jsonMetadata, verbose);
          assertReconstruction("question", questionStr, questionTok, -1, -1);
          putTokens(qaJson, questionTok);
          jsonMetadata.allTokens.addAll(questionTok.tokens);
          if (!hasAnswers) {
            numFullQuestions++;
            continue;
          }

          JSONArray answerObjs = (JSONArray) qaJson.get("answers");
          if (answerObjs.isEmpty()) {
            throw new RuntimeException("No answers\narticle:\n" + title + "\ncontext:\n" + contextStr + "\nid:\n" + idStr);
          }

          int numAnswers = answerObjs.size();
          int numInvalidAnswers = 0;
          for (Object answerObj : answerObjs) {
            JSONObject answerJson = (JSONObject) answerObj;

            String answerStr = (String) answerJson.get("text");
            int answerStartCharIdx = toIntExact((Long) answerJson.get("answer_start"));
            int answerAfterEndCharIdx = answerStartCharIdx + answerStr.length();
            int answerStartTokenIdx = contextTok.startCharIdxs.indexOf(answerStartCharIdx);
            int answerEndTokenIdx = contextTok.afterEndCharIdxs.indexOf(answerAfterEndCharIdx);
            if (verbose) {
              if (answerStartTokenIdx < 0) {
                print_bad_answer(true, idStr, contextTok, answerStr, answerStartCharIdx, answerAfterEndCharIdx);
              }
              if (answerEndTokenIdx < 0) {
                print_bad_answer(false, idStr, contextTok, answerStr, answerStartCharIdx, answerAfterEndCharIdx);
              }
            }
            if (answerStartTokenIdx < 0 || answerEndTokenIdx < 0) {
              answerJson.put("valid", false);
              numInvalidAnswers++;
            } else {
              assertReconstruction("answer", answerStr, contextTok, answerStartTokenIdx, answerEndTokenIdx);
              answerJson.put("valid", true);
              answerJson.put("start_token_idx", answerStartTokenIdx);
              answerJson.put("end_token_idx", answerEndTokenIdx);
            }
          }
          jsonMetadata.numAnswers += numAnswers;
          jsonMetadata.numInvalidAnswers += numInvalidAnswers;

          if (numInvalidAnswers == 0) {
            numFullQuestions++;
          } else if (numInvalidAnswers < numAnswers) {
            numPartialQuestions++;
          } else {
            numInvalidQuestions++;
          }
        }
        jsonMetadata.numQuestions += numQuestions;
        jsonMetadata.numFullQuestions += numFullQuestions;
        jsonMetadata.numPartialQuestions += numPartialQuestions;
        jsonMetadata.numInvalidQuestions += numInvalidQuestions;
        if (numInvalidQuestions == numQuestions) {
          numInvalidParagraphs++;
        }
      }
      jsonMetadata.numParagraphs += numParagraphs;
      jsonMetadata.numInvalidParagraphs += numInvalidParagraphs;
      if (numInvalidParagraphs == numParagraphs) {
        numInvalidArticles++;
      }
    }
    jsonMetadata.numArticles += numArticles;
    jsonMetadata.numInvalidArticles += numInvalidArticles;

    JSONArray unknownWordsArray = new JSONArray();
    unknownWordsArray.addAll(unknownWords);
    json.put("unknown_words", unknownWordsArray);

    jsonMetadata.unknownTokens.addAll(unknownWords);
    return jsonMetadata;
  }


  private static TokenizedText tokenize(
    String s, Set<String> knownWords, Set<String> unknownWords, boolean split, JsonMetadata jsonMetadata, boolean verbose) {

    String tokOpts = verbose ? TOK_OPTS.replace("untokenizable=noneKeep", "untokenizable=allKeep") : TOK_OPTS;
    PTBTokenizer<CoreLabel> tok = new PTBTokenizer<>(
      new StringReader(s), new CoreLabelTokenFactory(), TOK_OPTS);
    List<CoreLabel> coreLabels = tok.tokenize();

    TokenizedText tokenizedText = new TokenizedText();

    for (CoreLabel coreLabel : coreLabels) {
      String token = coreLabel.word();
      String original = coreLabel.originalText();
      String whitespaceAfter = coreLabel.after();
      int startCharIdx = coreLabel.beginPosition();
      int afterEndCharIdx = coreLabel.endPosition();

      // attempt matching to known word
      String knownWord = tokenToKnownWord(token, knownWords);
      if (knownWord != null) {
        tokenizedText.tokens.add(knownWord);
        tokenizedText.originals.add(original);
        tokenizedText.whitespaceAfters.add(whitespaceAfter);
        tokenizedText.startCharIdxs.add(startCharIdx);
        tokenizedText.afterEndCharIdxs.add(afterEndCharIdx);
        continue;
      }

      // attempt token splitting if configured to do so
      String[] subTokens = token.split("-");
      if (split && token.equals(original) && subTokens.length > 1 &&
        Arrays.stream(subTokens).noneMatch(subToken -> subToken.isEmpty())) {

        boolean useAlt = false;
        List<String> altTokens = new LinkedList<String>();
        List<String> altOriginals = new LinkedList<String>();
        List<String> altWhitespaceAfters = new LinkedList<String>();
        List<Integer> altStartCharIdxs = new LinkedList<Integer>();
        List<Integer> altAfterEndCharIdxs = new LinkedList<Integer>();
        Set<String> unknownWordsToAdd = new HashSet<String>();
        int altCharIdx = startCharIdx;
        for (int i=0; i<subTokens.length; i++) {
          String altToken = tokenToKnownWord(subTokens[i], knownWords);
          if (altToken == null) {
            altToken = subTokens[i];
            unknownWordsToAdd.add(altToken);
          } else {
            useAlt = true;
          }
          altTokens.add(altToken);
          altOriginals.add(subTokens[i]);
          altWhitespaceAfters.add(i == subTokens.length-1 ? whitespaceAfter: "");
          altStartCharIdxs.add(altCharIdx);
          altCharIdx += subTokens[i].length();
          altAfterEndCharIdxs.add(altCharIdx);
          if (i < subTokens.length-1) {
            altTokens.add("-");
            altOriginals.add("-");
            altWhitespaceAfters.add("");
            altStartCharIdxs.add(altCharIdx);
            altCharIdx += 1;
            altAfterEndCharIdxs.add(altCharIdx);
          }
        }
        if (altCharIdx != afterEndCharIdx) {
          throw new RuntimeException("Bad splitting");
        }
        if (useAlt) {
          tokenizedText.tokens.addAll(altTokens);
          tokenizedText.originals.addAll(altOriginals);
          tokenizedText.whitespaceAfters.addAll(altWhitespaceAfters);
          tokenizedText.startCharIdxs.addAll(altStartCharIdxs);
          tokenizedText.afterEndCharIdxs.addAll(altAfterEndCharIdxs);
          unknownWords.addAll(unknownWordsToAdd);
          jsonMetadata.numSplits++;
          if (verbose) {
            System.out.println(String.join("", altTokens));
          }
          continue;
        }
      }

      // not found
      tokenizedText.tokens.add(token);
      tokenizedText.originals.add(original);
      tokenizedText.whitespaceAfters.add(whitespaceAfter);
      tokenizedText.startCharIdxs.add(startCharIdx);
      tokenizedText.afterEndCharIdxs.add(afterEndCharIdx);
      unknownWords.add(token);
    }

    return tokenizedText;
  }


  private static String tokenToKnownWord(String token, Set<String> knownWords) {
    if (knownWords.contains(token)) {
      return token;
    }
    if (knownWords.contains(capitalize(token))) {
      return capitalize(token);
    }
    if (knownWords.contains(token.toLowerCase())) {
      return token.toLowerCase();
    }
    if (knownWords.contains(token.toUpperCase())) {
      return token.toUpperCase();
    }
    return null;
  }


  private static String capitalize(String s) {
    return Character.toUpperCase(s.charAt(0)) + s.substring(1);
  }


  private static void assertReconstruction(String name, String targetStr,
    TokenizedText tokenizedText, int reconstStartTokenIdx, int reconstEndTokenIdx) {
    if (reconstStartTokenIdx < 0 || reconstEndTokenIdx < 0) {
      reconstStartTokenIdx = 0;
      reconstEndTokenIdx = tokenizedText.tokens.size() - 1;
    }
    String reconstStr = tokenizedText.originals.get(reconstStartTokenIdx);
    for (int i=reconstStartTokenIdx+1; i<=reconstEndTokenIdx; i++) {
      reconstStr += tokenizedText.whitespaceAfters.get(i-1) + tokenizedText.originals.get(i);
    }
    if (!reconstStr.equals(targetStr.trim())) {
      throw new RuntimeException(
        "\nBad " + name + " reconstruction:\ntargetStr:\n[" + targetStr + "]\nreconstStr:\n[" + reconstStr + "]\n");
    }
  }


  @SuppressWarnings("unchecked")
  private static void putTokens(JSONObject jsonObj, TokenizedText tokenizedText) {
    JSONArray tokens = new JSONArray();
    JSONArray originals = new JSONArray();
    JSONArray whitespaceAfters = new JSONArray();
    tokens.addAll(tokenizedText.tokens);
    originals.addAll(tokenizedText.originals);
    whitespaceAfters.addAll(tokenizedText.whitespaceAfters);
    jsonObj.put("tokens", tokens);
    jsonObj.put("originals", originals);
    jsonObj.put("whitespace_afters", whitespaceAfters);
  }


  private static void print_bad_answer(boolean isStart, String idStr, TokenizedText contextTok,
    String answerStr, int answerStartCharIdx, int answerAfterEndCharIdx) {
    String msg = String.format(
      "\n%-4s%-20s%-20s%-9s%-9s",
      "idx", "word", "original", "beginPos", "endPos");
    List<Integer> contextCharIdxs = isStart ? contextTok.startCharIdxs : contextTok.afterEndCharIdxs;
    int answerCharIdx = isStart ? answerStartCharIdx : answerAfterEndCharIdx;

    int insertionIdx = -Collections.binarySearch(contextCharIdxs, answerCharIdx) - 1;
    int firstIdx = Math.max(0, insertionIdx - 3);
    int lastIdx = Math.min(contextCharIdxs.size() - 1, insertionIdx + 3);
    for (int i=firstIdx; i<=lastIdx; i++) {
      msg += String.format(
        "\n%-4d%-20s%-20s%-9d%-9d",
        i, "["+contextTok.tokens.get(i)+"]", "["+contextTok.originals.get(i)+"]",
        contextTok.startCharIdxs.get(i), contextTok.afterEndCharIdxs.get(i));
    }
    String msgTitle = String.format(
      "\nidStr: %s\nisStart: %b\nanswerStr: %s\nanswerStartCharIdx: %d\nanswerAfterEndCharIdx: %d\ninsertionIdx: %d\n",
      idStr, isStart, answerStr, answerStartCharIdx, answerAfterEndCharIdx, insertionIdx);
    msg += msgTitle;
    System.out.println(msg);
  }
}

