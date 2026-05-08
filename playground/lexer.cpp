#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/Lexer.h>
#include <clang/Basic/IdentifierTable.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/MemoryBuffer.h>

#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {

    if (argc < 2) {
        llvm::errs() << "Usage: tokenizer <file.cpp>\n";
        return 1;
    }

    // Read source file
    std::ifstream in(argv[1]);

    if (!in) {
        llvm::errs() << "Failed to open file\n";
        return 1;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();

    std::string code = buffer.str();

    // Create compiler instance
    clang::CompilerInstance ci;

    ci.createDiagnostics();

    ci.createFileManager();
    ci.createSourceManager(ci.getFileManager());

    auto& srcMgr = ci.getSourceManager();

    // Language options
    clang::LangOptions langOpts;

    langOpts.CPlusPlus = true;
    langOpts.CPlusPlus17 = true;

    // Create identifier table
    clang::IdentifierTable idTable(langOpts);

    // Create memory buffer
    auto memBuffer =
        llvm::MemoryBuffer::getMemBuffer(code);

    // Register buffer with source manager
    clang::FileID fid =
        srcMgr.createFileID(std::move(memBuffer));

    srcMgr.setMainFileID(fid);

    // Create raw lexer
    clang::Lexer lexer(
        fid,
        srcMgr.getBufferOrFake(fid),
        srcMgr,
        langOpts);

    clang::Token tok;

    // Tokenize loop
    while (!lexer.LexFromRawLexer(tok)) {

        // Raw lexer token kind
        llvm::StringRef rawKind =
            clang::tok::getTokenName(tok.getKind());

        // Token text
        std::string tokenText =
            clang::Lexer::getSpelling(
                tok,
                srcMgr,
                langOpts);

        std::string finalKind;

        // Convert raw_identifier -> actual keyword/identifier
        if (tok.is(clang::tok::raw_identifier)) {

            clang::IdentifierInfo& II =
                idTable.get(tokenText);

            clang::tok::TokenKind actualKind =
                II.getTokenID();

            finalKind =
                clang::tok::getTokenName(actualKind);

        } else {

            finalKind = rawKind.str();
        }

        llvm::outs()
            << "RAW = "
            << rawKind
            << " | FINAL = "
            << finalKind
            << " | TEXT = "
            << tokenText
            << "\n";
    }

    return 0;
}